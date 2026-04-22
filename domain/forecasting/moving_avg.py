from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Literal

import joblib
import numpy as np
import pandas as pd

from domain.forecasting.intervals import build_residual_interval

"""
이동평균(Moving Average) 기반 시계열 예측 베이스라인 모델.

이 파일의 책임:
- Simple MA / Seasonal MA 두 가지 모드 지원
- LGBMForecaster와 동일한 fit / predict_frame / forecast_recursive 인터페이스
- 잔차(Residual) 기반 90% 예측 구간 생성
- 모델 저장 / 로드

비교 목적:
  simple   → 직전 window 개 값의 평균. 가장 단순한 baseline.
  seasonal → 같은 시간대의 과거 window 주기 평균.
             IT 부하의 일간 주기성을 반영하므로 simple 대비 MAPE가 낮게 나옴.
"""

MAKind = Literal["simple", "seasonal"]


@dataclass
class MovingAverageMetadata:
    """
    모델과 함께 저장할 메타데이터.

    Attributes
    ----------
    target_name:
        예측 대상 컬럼명. 예: 'it_load_kw', 'cooling_demand_kw'
    kind:
        'simple' 또는 'seasonal'
    window:
        평균을 낼 기간 수
    seasonal_period:
        seasonal 모드에서 1주기의 길이 (예: 시간 단위 데이터에서 하루 = 24)
    model_type:
        모델 종류 식별자
    model_version:
        향후 버전 관리용 문자열
    """

    target_name: str
    kind: MAKind
    window: int
    seasonal_period: int
    model_type: str = "moving_average"
    model_version: str = "1.0.0"


class MovingAverageForecaster:
    """
    이동평균 기반 시계열 예측 베이스라인.

    두 가지 예측 모드
    -----------------
    simple
        직전 ``window`` 개 관측값의 단순 평균을 예측값으로 사용한다.
        추세나 계절성이 없는 경우에 적합하다.

    seasonal
        예측 시점과 같은 위상(phase)의 과거 ``window`` 개 관측값을 평균한다.
        예) 시간 단위 데이터, seasonal_period=24, window=7이면
            24 · 1일 전, 24 · 2일 전, …, 24 · 7일 전 값의 평균.
        IT 부하의 일간 반복 패턴을 포착하므로 simple 대비 MAPE가 낮게 나온다.

    인터페이스
    ----------
    LGBMForecaster와 동일한 fit / predict_frame / forecast_recursive 시그니처를
    유지하여 벤치마크 비교 코드 변경 없이 교체 가능하도록 설계한다.
    """

    def __init__(
        self,
        target_name: str,
        kind: MAKind = "seasonal",
        window: int = 7,
        seasonal_period: int = 24,
    ) -> None:
        """
        Parameters
        ----------
        target_name:
            예측 대상 컬럼명
        kind:
            'simple' 또는 'seasonal'
        window:
            평균을 낼 기간 수
            - simple 모드: 직전 window 개 스텝
            - seasonal 모드: 같은 위상의 과거 window 개 주기
        seasonal_period:
            seasonal 모드에서 1주기 길이.
            5분 단위 → 하루 = 288, 시간 단위 → 하루 = 24
        """
        if kind not in ("simple", "seasonal"):
            raise ValueError(f"kind must be 'simple' or 'seasonal', got '{kind}'.")
        if window < 1:
            raise ValueError("window must be >= 1.")
        if seasonal_period < 1:
            raise ValueError("seasonal_period must be >= 1.")

        self.target_name = target_name
        self.kind = kind
        self.window = window
        self.seasonal_period = seasonal_period

        self._history: np.ndarray = np.array([], dtype=float)
        self._residuals: np.ndarray | None = None
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # 공개 인터페이스
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame | None = None,
        series: pd.Series | np.ndarray | None = None,
        compute_residuals: bool = True,
    ) -> "MovingAverageForecaster":
        """
        학습 데이터를 받아 이력을 저장하고, 예측 구간 생성용 잔차를 계산한다.

        Parameters
        ----------
        train_df:
            ``target_name`` 컬럼이 포함된 DataFrame. series보다 우선 사용된다.
        series:
            target 값의 1차원 배열 또는 Series.
        compute_residuals:
            True이면 in-sample 잔차를 계산하여 저장한다.
            예측 구간 생성에 사용된다.

        Returns
        -------
        MovingAverageForecaster
            자기 자신(self)
        """
        if train_df is not None:
            if self.target_name not in train_df.columns:
                raise ValueError(
                    f"Target column '{self.target_name}' is missing from train_df."
                )
            values = train_df[self.target_name].to_numpy(dtype=float)
        elif series is not None:
            values = np.asarray(series, dtype=float)
        else:
            raise ValueError("Either train_df or series must be provided.")

        if values.ndim != 1:
            raise ValueError("Target series must be 1-dimensional.")

        self._history = values
        self.is_fitted = True

        if compute_residuals:
            self._residuals = self._compute_in_sample_residuals(values)

        return self

    def predict_frame(
        self,
        horizon: int,
        alpha: float = 0.1,
        timestamp_index: pd.DatetimeIndex | None = None,
        prediction_col: str | None = None,
    ) -> pd.DataFrame:
        """
        이력 끝부터 ``horizon`` 스텝 앞까지의 예측 결과를 DataFrame으로 반환한다.

        Parameters
        ----------
        horizon:
            예측 스텝 수 (예: 24 → 24시간 ahead)
        alpha:
            신뢰 구간 유의 수준. 0.1 → 90% 구간
        timestamp_index:
            결과 DataFrame에 추가할 타임스탬프 인덱스
        prediction_col:
            예측값 컬럼명. 기본값은 ``predicted_{target_name}``

        Returns
        -------
        pd.DataFrame
            컬럼: [timestamp(optional), predicted_{target_name}, lower_90, upper_90]
        """
        self._require_fitted()

        preds = self._forecast(self._history, horizon)
        prediction_col = prediction_col or f"predicted_{self.target_name}"

        result: dict[str, Any] = {prediction_col: preds}

        if self._residuals is not None and len(self._residuals) > 0:
            lower, upper = build_residual_interval(preds, self._residuals, alpha=alpha)
            result["lower_90"] = lower
            result["upper_90"] = upper

        df = pd.DataFrame(result)

        if timestamp_index is not None:
            df.index = timestamp_index[:horizon]

        return df

    def forecast_recursive(
        self,
        history_df: pd.DataFrame,
        horizon: int,
        make_next_feature_row: Callable[..., Any] | None = None,
        timestamp_col: str | None = None,
    ) -> pd.DataFrame:
        """
        재귀적 multi-step forecast를 수행한다.

        LGBMForecaster.forecast_recursive와 동일한 시그니처로,
        벤치마크 비교 루프에서 교체 없이 사용할 수 있다.

        이동평균 모델은 feature engineering이 필요 없으므로
        ``make_next_feature_row`` 인자는 무시된다.

        Parameters
        ----------
        history_df:
            ``target_name`` 컬럼이 포함된 과거 이력 DataFrame
        horizon:
            예측 스텝 수
        make_next_feature_row:
            LGBMForecaster 호환용. 이동평균 모델에서는 사용하지 않는다.
        timestamp_col:
            결과에 포함할 타임스탬프 컬럼명

        Returns
        -------
        pd.DataFrame
            컬럼: [step, timestamp(optional), predicted_{target_name}]
        """
        self._require_fitted()

        if self.target_name not in history_df.columns:
            raise ValueError(
                f"Target column '{self.target_name}' is missing from history_df."
            )
        if horizon <= 0:
            raise ValueError("horizon must be greater than 0.")

        series = history_df[self.target_name].to_numpy(dtype=float)
        preds = self._forecast(series, horizon)

        rows: list[dict[str, Any]] = []
        for step, pred in enumerate(preds, start=1):
            row: dict[str, Any] = {
                "step": step,
                f"predicted_{self.target_name}": pred,
            }

            if timestamp_col is not None and timestamp_col in history_df.columns:
                last_ts = history_df[timestamp_col].iloc[-1]
                # timestamp 컬럼이 datetime이면 step만큼 freq를 추정해 더한다.
                try:
                    freq = pd.infer_freq(history_df[timestamp_col])
                    offset = pd.tseries.frequencies.to_offset(freq)
                    row[timestamp_col] = last_ts + step * offset
                except Exception:
                    row[timestamp_col] = None

            rows.append(row)

        columns: list[str] = ["step"]
        if timestamp_col is not None:
            columns.append(timestamp_col)
        columns.append(f"predicted_{self.target_name}")

        return pd.DataFrame(rows)[columns]

    def save(self, file_path: str | Path) -> Path:
        """
        모델 이력 및 메타데이터를 joblib 파일로 저장한다.
        """
        self._require_fitted()

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "history": self._history,
            "residuals": self._residuals,
            "metadata": asdict(
                MovingAverageMetadata(
                    target_name=self.target_name,
                    kind=self.kind,
                    window=self.window,
                    seasonal_period=self.seasonal_period,
                )
            ),
        }
        joblib.dump(artifact, path)
        return path

    @classmethod
    def load(cls, file_path: str | Path) -> "MovingAverageForecaster":
        """
        저장된 joblib 파일에서 모델을 로드한다.
        """
        path = Path(file_path)
        artifact = joblib.load(path)
        metadata = artifact["metadata"]

        forecaster = cls(
            target_name=metadata["target_name"],
            kind=metadata["kind"],
            window=metadata["window"],
            seasonal_period=metadata["seasonal_period"],
        )
        forecaster._history = artifact["history"]
        forecaster._residuals = artifact["residuals"]
        forecaster.is_fitted = True
        return forecaster

    # ------------------------------------------------------------------
    # 내부 로직
    # ------------------------------------------------------------------

    def _forecast(self, series: np.ndarray, horizon: int) -> np.ndarray:
        """
        ``series`` 끝부터 ``horizon`` 스텝의 예측값 배열을 반환한다.

        simple 모드에서는 예측값을 이력에 누적하며 다음 스텝을 예측한다.
        seasonal 모드에서는 이력 길이가 고정되어 있으므로 누적 불필요.
        """
        if self.kind == "simple":
            return self._forecast_simple(series, horizon)
        return self._forecast_seasonal(series, horizon)

    def _forecast_simple(self, series: np.ndarray, horizon: int) -> np.ndarray:
        """
        직전 window 개 관측값의 평균을 재귀적으로 예측한다.

        예측값을 이력에 추가해 다음 스텝 예측에 사용하므로
        horizon이 길어질수록 예측값은 평균으로 수렴한다.
        """
        buf = list(series[-self.window :])
        preds = np.empty(horizon, dtype=float)

        for i in range(horizon):
            pred = float(np.mean(buf[-self.window :]))
            preds[i] = pred
            buf.append(pred)

        return preds

    def _forecast_seasonal(self, series: np.ndarray, horizon: int) -> np.ndarray:
        """
        같은 위상(phase)의 과거 window 개 주기 관측값의 평균을 예측값으로 사용한다.

        step h (1-indexed) 의 예측값 =
            mean(series[-(seasonal_period * k) + (h - 1)]  for k in 1..window)

        필요한 이력이 부족하면 존재하는 범위만 사용한다.
        """
        n = len(series)
        preds = np.empty(horizon, dtype=float)

        for h in range(1, horizon + 1):
            # 예측 시점과 같은 위상의 과거 인덱스 수집
            indices = [
                n - self.seasonal_period * k + (h - 1)
                for k in range(1, self.window + 1)
            ]
            valid = [idx for idx in indices if 0 <= idx < n]

            if valid:
                preds[h - 1] = float(np.mean(series[valid]))
            else:
                # fallback: 마지막 window 개 평균
                preds[h - 1] = float(np.mean(series[-self.window :]))

        return preds

    def _compute_in_sample_residuals(self, series: np.ndarray) -> np.ndarray:
        """
        학습 데이터에 대한 in-sample 잔차(실제 - 예측)를 계산한다.

        각 시점 t의 예측은 t 이전 이력만 사용하므로 data leakage가 없다.
        잔차는 ``build_residual_interval``에 전달되어 예측 구간을 생성한다.
        """
        min_len = (
            self.window if self.kind == "simple"
            else self.seasonal_period * self.window
        )

        if len(series) <= min_len:
            return np.array([], dtype=float)

        residuals: list[float] = []

        for t in range(min_len, len(series)):
            pred = float(self._forecast(series[:t], horizon=1)[0])
            residuals.append(series[t] - pred)

        return np.array(residuals, dtype=float)

    def _require_fitted(self) -> None:
        """
        모델이 학습되었는지 확인한다.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Model is not fitted. Call fit() first."
            )
