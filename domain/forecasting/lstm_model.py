from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

"""
LSTM 기반 시계열 예측 모델 로직.

이 파일에서 구현할 것
- PyTorch LSTM 모델 정의
- sequence 데이터 학습 루프
- validation / loss 계산
- multi-step forecast inference
- 모델 저장/로드 인터페이스

이 파일의 책임
- 시계열 sequence 모델 정의와 학습
- 입력 shape / hidden state 처리
- horizon 단위 예측 로직 제공
"""


DEFAULT_LSTM_PARAMS: dict[str, Any] = {
    "hidden_size": 64,      # LSTM 셀의 기억 용량
    "num_layers": 2,        # LSTM 층의 높이
    "dropout": 0.1,         # 과적합 방지를 위해, 10%의 뉴런을 끔
    "learning_rate": 1e-3,  # 가중치를 한 번에 얼마나 수정할지
    "batch_size": 32,       # 한 번에 공부하는 양. 전체 데이터를 32개씩 묶어서 학습
    "epochs": 50,           # 전체 데이터 반복 횟수.
    "device": "cpu",        # 연산 장치
}


@dataclass
class LSTMModelMetadata:
    """
    모델과 함께 저장할 메타데이터.
    """

    target_name: str
    feature_columns: list[str]      # 학습에 사용된 feature들의 이름과 순서
    sequence_length: int            # sequence_length = 24는 최근 24개 시점을 보고 다음 1개를 예측함.
    model_type: str = "lstm_regressor"  
    model_version: str = "1.0.0"
    params: dict[str, Any] = field(default_factory=dict)


class SequenceDataset(Dataset):
    """
    LSTM 학습용 시퀀스 데이터셋.
    메모리에 있는 NumPy 배열 데이터를 PyTorch가 이해할 수 있는 텐서(Tensor) 형식으로 변환하고, 
    모델이 요청할 때마다 데이터를 하나씩 넘겨주는 역할
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) != len(y): # 데이터 개수가 일치하는지 체크
            raise ValueError("X and y must have the same number of samples.")

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    """
    기본 LSTM 회귀 모델.
    입력 shape: (batch, seq_len, feature_dim)
    출력 shape: (batch, 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_size: int = 1,
    ) -> None:
        super().__init__()

        effective_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden)


class LSTMForecaster:
    """
    LSTM 기반 시계열 예측기.

    역할
    ----
    - 시계열 window 생성
    - 모델 학습 / 검증
    - 단일/배치 추론
    - 재귀적 multi-step forecast
    - 모델 저장 및 로드

    Notes
    -----
    - 입력 데이터는 최소한 feature_columns와 target_name 컬럼을 가져야 한다.
    - predict(df)는 df 내부에서 sliding window를 만들어 예측한다.
    - 따라서 예측 결과는 원본 df의 앞 sequence_length개 구간 이후부터 생성된다.
    """

    def __init__(
        self,
        target_name: str,
        feature_columns: Sequence[str],
        sequence_length: int = 24,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than 0.")
        if not feature_columns:
            raise ValueError("feature_columns must not be empty.")

        self.target_name = target_name
        self.feature_columns = list(feature_columns)
        self.sequence_length = sequence_length
        self.params: dict[str, Any] = {**DEFAULT_LSTM_PARAMS, **(params or {})}

        self.device = torch.device(self.params.get("device", "cpu"))

        self.model = LSTMRegressor(
            input_size=len(self.feature_columns),
            hidden_size=int(self.params["hidden_size"]),
            num_layers=int(self.params["num_layers"]),
            dropout=float(self.params["dropout"]),
        ).to(self.device)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.params["learning_rate"]),
        )

        self.is_fitted: bool = False
        self.train_loss_history: list[float] = []
        self.valid_loss_history: list[float] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        shuffle: bool = True,
    ) -> "LSTMForecaster":
        """
        모델을 학습한다.
        """
        self._validate_train_df(train_df)

        epochs = int(epochs or self.params["epochs"])
        batch_size = int(batch_size or self.params["batch_size"])

        X_train, y_train = self._build_sequences_from_frame(train_df)
        if len(X_train) == 0:
            raise ValueError(
                "Not enough rows to create training sequences. "
                "Increase train_df length or reduce sequence_length."
            )

        train_loader = DataLoader(
            SequenceDataset(X_train, y_train),
            batch_size=batch_size,
            shuffle=shuffle,
        )

        valid_loader: DataLoader | None = None
        if valid_df is not None:
            self._validate_train_df(valid_df)
            X_valid, y_valid = self._build_sequences_from_frame(valid_df)
            if len(X_valid) > 0:
                valid_loader = DataLoader(
                    SequenceDataset(X_valid, y_valid),
                    batch_size=batch_size,
                    shuffle=False,
                )

        self.train_loss_history.clear()
        self.valid_loss_history.clear()

        for _ in range(epochs):
            train_loss = self._train_one_epoch(train_loader)
            self.train_loss_history.append(train_loss)

            if valid_loader is not None:
                valid_loss = self._evaluate(valid_loader)
                self.valid_loss_history.append(valid_loss)

        self.is_fitted = True
        return self

    def predict_sequences(self, X_seq: np.ndarray) -> np.ndarray:
        """
        이미 만들어진 시퀀스 입력(batch, seq_len, feature_dim)에 대해 예측한다.
        """
        self._require_fitted()

        if X_seq.ndim != 3:
            raise ValueError("X_seq must be a 3D array: (batch, seq_len, feature_dim).")
        if X_seq.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected seq_len={self.sequence_length}, got {X_seq.shape[1]}."
            )
        if X_seq.shape[2] != len(self.feature_columns):
            raise ValueError(
                f"Expected feature_dim={len(self.feature_columns)}, got {X_seq.shape[2]}."
            )

        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_seq, dtype=torch.float32, device=self.device)
            preds = self.model(x_tensor).squeeze(-1).cpu().numpy()

        return preds.astype(float)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        DataFrame에서 sliding window를 생성해 예측한다.
        """
        self._validate_feature_df(df)

        X_seq = self._build_input_sequences(df)
        if len(X_seq) == 0:
            raise ValueError(
                "Not enough rows to create prediction sequences. "
                "Input rows must be greater than sequence_length."
            )

        return self.predict_sequences(X_seq)

    def predict_frame(
        self,
        df: pd.DataFrame,
        timestamp_col: str | None = None,
        prediction_col: str | None = None,
    ) -> pd.DataFrame:
        """
        예측 결과를 DataFrame으로 반환한다.
        """
        prediction_col = prediction_col or f"predicted_{self.target_name}"
        preds = self.predict(df)

        result = pd.DataFrame({prediction_col: preds})

        if timestamp_col is not None and timestamp_col in df.columns:
            aligned_ts = df.iloc[self.sequence_length:][timestamp_col].reset_index(drop=True)
            result.insert(0, timestamp_col, aligned_ts)

        return result

    def forecast_recursive(
        self,
        history_df: pd.DataFrame,
        horizon: int,
        make_next_feature_row: Callable[[pd.DataFrame, int], pd.DataFrame | pd.Series],
        timestamp_col: str | None = None,
    ) -> pd.DataFrame:
        """
        재귀적 multi-step forecast를 수행한다.

        작동 방식
        --------
        1. 최근 sequence_length 구간을 입력으로 다음 시점 예측
        2. 예측값을 history에 붙임
        3. 다음 시점 예측 반복
        """
        self._require_fitted()
        self._validate_train_df(history_df)

        if horizon <= 0:
            raise ValueError("horizon must be greater than 0.")
        if len(history_df) < self.sequence_length:
            raise ValueError(
                "history_df must contain at least sequence_length rows for recursive forecast."
            )

        simulated_history = history_df.copy()
        rows: list[dict[str, Any]] = []

        for step in range(1, horizon + 1):
            next_features = make_next_feature_row(simulated_history, step)

            if isinstance(next_features, pd.Series):
                next_features = next_features.to_frame().T

            if not isinstance(next_features, pd.DataFrame):
                raise TypeError(
                    "make_next_feature_row must return a pandas DataFrame or Series."
                )

            if len(next_features) != 1:
                raise ValueError("make_next_feature_row must return exactly one row.")

            next_features = next_features.copy()

            missing = [col for col in self.feature_columns if col not in next_features.columns]
            if missing:
                raise ValueError(f"Missing feature columns in next row: {missing}")

            window = pd.concat(
                [
                    simulated_history[self.feature_columns].iloc[-(self.sequence_length - 1):],
                    next_features[self.feature_columns],
                ],
                ignore_index=True,
            )

            x_seq = window.to_numpy(dtype=np.float32).reshape(1, self.sequence_length, -1)
            pred = float(self.predict_sequences(x_seq)[0])

            row: dict[str, Any] = {
                "step": step,
                f"predicted_{self.target_name}": pred,
            }

            if timestamp_col is not None and timestamp_col in next_features.columns:
                row[timestamp_col] = next_features.iloc[0][timestamp_col]

            rows.append(row)

            appended_row = next_features.copy()
            appended_row[self.target_name] = pred
            simulated_history = pd.concat([simulated_history, appended_row], ignore_index=True)

        result_df = pd.DataFrame(rows)

        columns = ["step"]
        if timestamp_col is not None and timestamp_col in result_df.columns:
            columns.append(timestamp_col)
        columns.append(f"predicted_{self.target_name}")

        return result_df[columns]

    def save(self, file_path: str | Path) -> Path:
        """
        모델과 메타데이터를 저장한다.
        """
        self._require_fitted()

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "state_dict": self.model.state_dict(),
            "metadata": asdict(
                LSTMModelMetadata(
                    target_name=self.target_name,
                    feature_columns=self.feature_columns,
                    sequence_length=self.sequence_length,
                    params=self.params,
                )
            ),
            "train_loss_history": self.train_loss_history,
            "valid_loss_history": self.valid_loss_history,
        }

        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, file_path: str | Path) -> "LSTMForecaster":
        """
        저장된 파일에서 모델을 로드한다.
        """
        path = Path(file_path)
        payload = torch.load(path, map_location="cpu")

        metadata = payload["metadata"]
        params = metadata.get("params", {})

        forecaster = cls(
            target_name=metadata["target_name"],
            feature_columns=metadata["feature_columns"],
            sequence_length=metadata["sequence_length"],
            params=params,
        )
        forecaster.model.load_state_dict(payload["state_dict"])
        forecaster.model.to(forecaster.device)

        forecaster.train_loss_history = list(payload.get("train_loss_history", []))
        forecaster.valid_loss_history = list(payload.get("valid_loss_history", []))
        forecaster.is_fitted = True
        return forecaster

    def _train_one_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()

        total_loss = 0.0
        total_count = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(x_batch)
            loss = self.loss_fn(pred, y_batch)
            loss.backward()
            self.optimizer.step()

            batch_size = len(x_batch)
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

        return total_loss / total_count

    def _evaluate(self, data_loader: DataLoader) -> float:
        self.model.eval()

        total_loss = 0.0
        total_count = 0

        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                pred = self.model(x_batch)
                loss = self.loss_fn(pred, y_batch)

                batch_size = len(x_batch)
                total_loss += float(loss.item()) * batch_size
                total_count += batch_size

        return total_loss / total_count

    def _build_sequences_from_frame(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        train_df에서 (X_seq, y) 형태의 sliding window 데이터를 만든다.
        """
        features = df[self.feature_columns].to_numpy(dtype=np.float32)
        target = df[self.target_name].to_numpy(dtype=np.float32)

        X_seq: list[np.ndarray] = []
        y_seq: list[float] = []

        for idx in range(self.sequence_length, len(df)):
            X_seq.append(features[idx - self.sequence_length:idx])
            y_seq.append(target[idx])

        return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)

    def _build_input_sequences(self, df: pd.DataFrame) -> np.ndarray:
        """
        예측용 DataFrame에서 입력 시퀀스만 생성한다.
        """
        features = df[self.feature_columns].to_numpy(dtype=np.float32)

        X_seq: list[np.ndarray] = []
        for idx in range(self.sequence_length, len(df)):
            X_seq.append(features[idx - self.sequence_length:idx])

        return np.asarray(X_seq, dtype=np.float32)

    def _validate_train_df(self, df: pd.DataFrame) -> None:
        if self.target_name not in df.columns:
            raise ValueError(f"Target column '{self.target_name}' is missing.")

        missing = [col for col in self.feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

    def _validate_feature_df(self, df: pd.DataFrame) -> None:
        missing = [col for col in self.feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns for prediction: {missing}")

        if len(df) <= self.sequence_length:
            raise ValueError(
                "Prediction input must contain more rows than sequence_length."
            )

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")
