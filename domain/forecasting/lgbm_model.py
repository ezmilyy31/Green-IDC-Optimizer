from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

"""
LightGBM 기반 시계열 예측 모델 로직.

이 파일에서 구현할 것
- LightGBM regressor 생성
- 학습 데이터(X, y) 기반 모델 학습
- IT 부하 예측
- 냉각 수요 예측
- 재귀적 multi-step forecast
- feature importance 확인

이 파일의 책임
- 모델 자체의 fit / predict 로직
- 모델 하이퍼파라미터 관리
- 학습/추론 인터페이스 제공
"""

# LightGBM이 모델을 학습할 때 사용하는 Hyperparameters 집합.
DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "regression",  # 학습목표: 회귀
    "boosting_type": "gbdt",    # Gradient Boosting Decision Tree
    "n_estimators": 500,        # tree 개수. 학습을 몇 단계 반복할지 결정. 너무 크면 학습 시간 증가 & overfitting
    "learning_rate": 0.05,      
    "num_leaves": 31,           # 모델복잡도 결정
    "max_depth": -1,            # 트리 깊이 제한 (-1: 제한 없음)
    "subsample": 0.8,           # 전체 데이터의 80%만 무작위로 사용하여 트리 생성
    "colsample_bytree": 0.8,    # 전체 feature 중 80%만 선택해서 사용
    "random_state": 42,         # 난수 시드 고정
    "n_jobs": -1,               # 병렬 처리. 사용가능한 모든 CPU 코어를 학습에 투입
}


@dataclass
class LGBMModelMetadata:
    """
    모델과 함께 저장할 메타데이터.

    Attributes
    ----------
    target_name:
        예측 대상 컬럼명. 예: 'it_load_kw', 'cooling_demand_kw'
    feature_columns:
        학습 및 추론 시 사용하는 feature 컬럼 목록
    model_type:
        모델 종류 식별자
    model_version:
        향후 버전 관리용 문자열
    """

    target_name: str
    feature_columns: list[str]
    model_type: str = "lightgbm_regressor"
    model_version: str = "1.0.0"

class LGBMForecaster:
    """
    LightGBM 기반 시계열/회귀 예측기.

    이 클래스의 역할
    ----------------
    - 전처리 완료된 feature 데이터로 모델 학습
    - 단일/배치 추론
    - feature importance 조회
    - 모델 저장 및 로드
    - 필요 시 재귀적 multi-step forecast 보조

    Notes
    -----
    - 입력 데이터는 이미 feature engineering이 끝났다고 가정한다.
    - recursive forecasting에서 다음 시점 feature를 어떻게 만들지는
      외부 callback(make_next_feature_row)으로 위임한다.
    """

    def __init__(
        self,
        target_name: str,                               # 맞춰야 할 정답. IT 부하 예측 / 냉각 수요 예측
        feature_columns: Sequence[str] | None = None,   
        params: Mapping[str, Any] | None = None,        # 앞서 정의한 `DEFAULT_LGBM_PARAMS`가 여기에 들어감.
    ) -> None:
        self.target_name = target_name
        self.feature_columns = list(feature_columns) if feature_columns is not None else []
        self.params: dict[str, Any] = {**DEFAULT_LGBM_PARAMS, **(params or {})}
        self.model = lgb.LGBMRegressor(**self.params)
        self.is_fitted: bool = False

    def fit( 
        self,
        train_df: pd.DataFrame | None = None,
        X: pd.DataFrame | None = None,
        y: pd.Series | np.ndarray | None = None,
        valid_df: pd.DataFrame | None = None,
        eval_set: list[tuple[pd.DataFrame, pd.Series | np.ndarray]] | None = None,
        categorical_features: Sequence[str] | None = None,

        # TODO: 과적합 방지를 위한 early_stopping 내용 추가해야 함.
    ) -> "LGBMForecaster":
        """
        모델을 학습한다.

        Parameters
        ----------
        train_df:
            feature와 target이 모두 들어 있는 학습용 DataFrame
        X:
            feature DataFrame
        y:
            target Series 또는 ndarray
        valid_df:
            validation용 DataFrame (train_df 방식 사용 시)
        eval_set:
            LightGBM eval_set 형식 [(X_valid, y_valid), ...]
        categorical_features:
            범주형 feature 이름 목록

        Returns
        -------
        LGBMForecaster
            자기 자신(self)
        """
        if train_df is not None: # 유연성
            self._validate_train_df(train_df)
            X_train = train_df[self.feature_columns].copy()
            y_train = train_df[self.target_name].copy()
        else:
            if X is None or y is None:
                raise ValueError("Either train_df or both X and y must be provided.")
            if not isinstance(X, pd.DataFrame):
                raise TypeError("X must be a pandas DataFrame.")
            X_train = X.copy()
            y_train = pd.Series(y, name=self.target_name)

            if not self.feature_columns:
                self.feature_columns = list(X_train.columns)

        fit_kwargs: dict[str, Any] = {}
        callbacks = []

        if valid_df is not None: # 검증
            self._validate_train_df(valid_df)
            fit_kwargs["eval_set"] = [
                (valid_df[self.feature_columns], valid_df[self.target_name])
            ]
            # 검증 셋이 있으면 50번 동안 성능 향상이 없을 때 조기 종료
            callbacks.append(early_stopping(stopping_rounds=50))
            callbacks.append(log_evaluation(period=100)) # 100번마다 로그 출력

        elif eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
            callbacks.append(early_stopping(stopping_rounds=50))
            callbacks.append(log_evaluation(period=100))

        if categorical_features is not None:
            fit_kwargs["categorical_feature"] = list(categorical_features)
        
        if callbacks:
            fit_kwargs["callbacks"] = callbacks 

        self.model.fit(X_train[self.feature_columns], y_train, **fit_kwargs)
        self.is_fitted = True   # 학습 완료 표시
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        입력 feature DataFrame에 대해 예측값을 반환한다.
        """
        self._require_fitted()
        X_prepared = self._prepare_feature_frame(X)
        preds = self.model.predict(X_prepared)
        return np.asarray(preds, dtype=float)

    def predict_frame(
        self,
        X: pd.DataFrame,
        timestamp_col: str | None = None,
        prediction_col: str | None = None,
    ) -> pd.DataFrame:
        """
        예측 결과를 DataFrame으로 반환한다.

        Parameters
        ----------
        X:
            feature DataFrame
        timestamp_col:
            포함하고 싶은 timestamp 컬럼명
        prediction_col:
            예측 컬럼명. 기본값은 f'predicted_{target_name}'

        Returns
        -------
        pd.DataFrame
        """
        prediction_col = prediction_col or f"predicted_{self.target_name}"
        preds = self.predict(X)

        result = pd.DataFrame({prediction_col: preds})

        if timestamp_col is not None and timestamp_col in X.columns:
            result.insert(0, timestamp_col, X[timestamp_col].values)

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

        작동 방식 추가 설명.
        1. 현재 데이터로 t+1 시점 예측
        2. 예측 값을 실제 데이터인 것처럼 데이터셋에 합침
        3. 합쳐진 데이터로 t+2 시점 예측

        Parameters
        ----------
        history_df:
            과거 이력 데이터. target 컬럼이 포함되어 있어야 한다.
        horizon:
            예측 step 수
        make_next_feature_row:
            현재까지의 simulated history와 step index를 받아
            '다음 시점 1행 feature'를 반환하는 callback
        timestamp_col:
            결과에 함께 담고 싶은 timestamp 컬럼명

        Returns
        -------
        pd.DataFrame
            각 step의 예측 결과 테이블

        Notes
        -----
        lag/rolling feature를 어떤 방식으로 갱신할지는 프로젝트마다 다르므로,
        이 함수는 feature 생성 자체를 하지 않고 외부 callback에 위임한다.
        """
        self._require_fitted()

        if horizon <= 0:
            raise ValueError("horizon must be greater than 0.")

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
            pred = float(self.predict(next_features)[0])

            row: dict[str, Any] = {
                "step": step,
                f"predicted_{self.target_name}": pred,
            }

            if timestamp_col is not None and timestamp_col in next_features.columns:
                row[timestamp_col] = next_features.iloc[0][timestamp_col]

            if "outdoor_temp_c" in next_features.columns:
                row["outdoor_temp_c"] = next_features.iloc[0]["outdoor_temp_c"]

            rows.append(row)

            appended_row = next_features.copy()
            appended_row[self.target_name] = pred
            simulated_history = pd.concat(
                [simulated_history, appended_row], ignore_index=True
            )

        columns: list[str] = ["step"]
        if timestamp_col is not None:
            columns.append(timestamp_col)
        columns.append(f"predicted_{self.target_name}")
        
        if "outdoor_temp_c" in pd.DataFrame(rows).columns:
            columns.append("outdoor_temp_c")

        return pd.DataFrame(rows)[columns]
    

    def get_feature_importance(
        self,
        importance_type: str = "gain",
        normalize: bool = False,
    ) -> pd.DataFrame:
        """
        feature importance를 DataFrame으로 반환한다.

        Parameters
        ----------
        importance_type:
            'gain' 또는 'split'
        normalize:
            True면 importance 합이 1이 되도록 정규화
        """
        self._require_fitted()

        booster = self.model.booster_
        if booster is None:
            raise RuntimeError("LightGBM booster is not available.")

        importance = booster.feature_importance(importance_type=importance_type)
        importance = importance.astype(float)

        if normalize:
            total = importance.sum()
            if total > 0:
                importance = importance / total

        df = pd.DataFrame(
            {
                "feature": self.feature_columns,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False, ignore_index=True)

        return df

    def save(self, file_path: str | Path) -> Path:
        """
        모델과 메타데이터를 joblib 파일로 저장한다. -> 나중에 다시 사용할 수 있도록
        """
        self._require_fitted()

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model": self.model,
            "params": self.params,
            "metadata": asdict(
                LGBMModelMetadata(
                    target_name=self.target_name,
                    feature_columns=self.feature_columns,
                )
            ),
        }
        joblib.dump(artifact, path)
        return path

    @classmethod
    def load(cls, file_path: str | Path) -> "LGBMForecaster":
        """
        저장된 joblib 파일에서 모델을 로드한다.
        """
        path = Path(file_path)
        artifact = joblib.load(path)

        metadata = artifact["metadata"]
        params = artifact["params"]
        model = artifact["model"]

        forecaster = cls(
            target_name=metadata["target_name"],
            feature_columns=metadata["feature_columns"],
            params=params,
        )
        forecaster.model = model
        forecaster.is_fitted = True
        return forecaster

    def _validate_train_df(self, df: pd.DataFrame) -> None:
        """
        train/valid DataFrame에 필요한 컬럼이 있는지 검증한다.
        """
        if self.target_name not in df.columns:
            raise ValueError(f"Target column '{self.target_name}' is missing.")

        if not self.feature_columns:
            self.feature_columns = [col for col in df.columns if col != self.target_name]

        missing = [col for col in self.feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

    def _prepare_feature_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        예측용 feature DataFrame을 검증하고 정렬한다.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if not self.feature_columns:
            raise RuntimeError("feature_columns is empty. Fit or load a model first.")

        missing = [col for col in self.feature_columns if col not in X.columns]
        if missing:
            raise ValueError(f"Missing feature columns for prediction: {missing}")

        return X[self.feature_columns].copy()

    def _require_fitted(self) -> None:
        """
        모델이 학습 또는 로드되었는지 확인한다.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() or load() first.")