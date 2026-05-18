"""
domain/forecasting/moving_avg.py 단위 테스트.

대상: MovingAverageForecaster — simple / seasonal 두 가지 예측 모드.
"""
import numpy as np
import pandas as pd
import pytest

from domain.forecasting.moving_avg import MovingAverageForecaster


class TestInit:
    def test_defaults(self):
        m = MovingAverageForecaster("val")
        assert m.kind == "seasonal"
        assert m.window == 7
        assert m.seasonal_period == 24
        assert m.is_fitted is False

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="kind must be"):
            MovingAverageForecaster("val", kind="exponential")

    def test_window_zero_raises(self):
        with pytest.raises(ValueError, match="window must be >= 1"):
            MovingAverageForecaster("val", window=0)

    def test_seasonal_period_zero_raises(self):
        with pytest.raises(ValueError, match="seasonal_period must be >= 1"):
            MovingAverageForecaster("val", seasonal_period=0)


class TestFit:
    def test_fit_with_series_sets_is_fitted(self):
        m = MovingAverageForecaster("val", kind="simple", window=3)
        m.fit(series=np.array([1., 2., 3., 4., 5.]))
        assert m.is_fitted is True

    def test_fit_with_train_df_sets_is_fitted(self):
        m = MovingAverageForecaster("val", kind="simple", window=3)
        df = pd.DataFrame({"val": [1., 2., 3., 4., 5.]})
        m.fit(train_df=df)
        assert m.is_fitted is True

    def test_fit_returns_self(self):
        m = MovingAverageForecaster("val", kind="simple", window=3)
        result = m.fit(series=np.array([1., 2., 3.]))
        assert result is m

    def test_fit_raises_when_nothing_provided(self):
        m = MovingAverageForecaster("val")
        with pytest.raises(ValueError, match="Either train_df or series"):
            m.fit()

    def test_fit_raises_when_target_missing_from_train_df(self):
        m = MovingAverageForecaster("val")
        df = pd.DataFrame({"other": [1., 2., 3.]})
        with pytest.raises(ValueError, match="Target column"):
            m.fit(train_df=df)

    def test_fit_raises_when_series_not_1d(self):
        m = MovingAverageForecaster("val")
        with pytest.raises(ValueError, match="1-dimensional"):
            m.fit(series=np.ones((3, 2)))

    def test_history_stored_correctly(self):
        m = MovingAverageForecaster("val", kind="simple", window=2)
        m.fit(series=np.array([1., 2., 3.]))
        assert np.array_equal(m._history, [1., 2., 3.])

    def test_compute_residuals_false_leaves_none(self):
        m = MovingAverageForecaster("val", kind="simple", window=2)
        m.fit(series=np.array([1., 2., 3., 4.]), compute_residuals=False)
        assert m._residuals is None

    def test_residuals_computed_when_long_series(self, seasonal_series):
        # window=2, period=4 → min_len=8. 12개 시리즈 → 잔차 4개
        m = MovingAverageForecaster("val", kind="seasonal", window=2, seasonal_period=4)
        m.fit(series=seasonal_series)
        assert m._residuals is not None
        assert len(m._residuals) == 4


class TestForecastSimple:
    def test_predict_frame_returns_dataframe(self, fitted_simple_ma):
        result = fitted_simple_ma.predict_frame(horizon=3)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_predict_frame_column_name_default(self, fitted_simple_ma):
        result = fitted_simple_ma.predict_frame(horizon=2)
        assert "predicted_val" in result.columns

    def test_predict_frame_custom_col(self, fitted_simple_ma):
        result = fitted_simple_ma.predict_frame(horizon=2, prediction_col="yhat")
        assert "yhat" in result.columns
        assert "predicted_val" not in result.columns

    def test_first_step_value(self, fitted_simple_ma):
        # series=[1,2,3,4,5], window=3 → buf=[3,4,5], 첫 예측 = mean([3,4,5]) = 4.0
        result = fitted_simple_ma.predict_frame(horizon=1)
        assert result["predicted_val"].iloc[0] == pytest.approx(4.0)

    def test_all_predictions_finite(self, fitted_simple_ma):
        result = fitted_simple_ma.predict_frame(horizon=5)
        assert np.isfinite(result["predicted_val"].to_numpy()).all()

    def test_interval_cols_present(self, fitted_simple_ma):
        # 잔차가 존재하므로 예측 구간 컬럼이 포함돼야 함
        result = fitted_simple_ma.predict_frame(horizon=3)
        assert "lower_90" in result.columns
        assert "upper_90" in result.columns

    def test_predict_frame_raises_when_not_fitted(self):
        m = MovingAverageForecaster("val")
        with pytest.raises(RuntimeError, match="not fitted"):
            m.predict_frame(horizon=2)


class TestForecastSeasonal:
    def test_predict_frame_horizon_4(self, fitted_seasonal_ma):
        result = fitted_seasonal_ma.predict_frame(horizon=4)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4

    def test_seasonal_phase_alignment(self, fitted_seasonal_ma):
        # series=tile([10,20,30,40],3), n=12, window=2, period=4
        # h=1: indices [8,4] → series[8]=10, series[4]=10 → mean=10.0
        # h=2: indices [9,5] → 20.0, h=3→30.0, h=4→40.0
        result = fitted_seasonal_ma.predict_frame(horizon=4)
        assert result["predicted_val"].tolist() == pytest.approx([10., 20., 30., 40.])

    def test_interval_cols_present(self, fitted_seasonal_ma):
        result = fitted_seasonal_ma.predict_frame(horizon=4)
        assert "lower_90" in result.columns
        assert "upper_90" in result.columns

    def test_interval_lower_leq_pred(self, fitted_seasonal_ma):
        result = fitted_seasonal_ma.predict_frame(horizon=4)
        assert (result["lower_90"].values <= result["predicted_val"].values).all()

    def test_interval_upper_geq_pred(self, fitted_seasonal_ma):
        result = fitted_seasonal_ma.predict_frame(horizon=4)
        assert (result["upper_90"].values >= result["predicted_val"].values).all()

    def test_timestamp_index_applied(self, fitted_seasonal_ma):
        ts_idx = pd.date_range("2024-01-01", periods=4, freq="h")
        result = fitted_seasonal_ma.predict_frame(horizon=4, timestamp_index=ts_idx)
        assert result.index[0] == pd.Timestamp("2024-01-01")
        assert result.index[3] == pd.Timestamp("2024-01-01 03:00:00")


class TestForecastRecursiveMA:
    def test_returns_dataframe_with_correct_len(self, fitted_seasonal_ma, seasonal_series):
        history_df = pd.DataFrame({"val": seasonal_series})
        result = fitted_seasonal_ma.forecast_recursive(history_df, horizon=3)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_step_column_sequential(self, fitted_seasonal_ma, seasonal_series):
        history_df = pd.DataFrame({"val": seasonal_series})
        result = fitted_seasonal_ma.forecast_recursive(history_df, horizon=3)
        assert result["step"].tolist() == [1, 2, 3]

    def test_raises_for_zero_horizon(self, fitted_seasonal_ma, seasonal_series):
        history_df = pd.DataFrame({"val": seasonal_series})
        with pytest.raises(ValueError, match="greater than 0"):
            fitted_seasonal_ma.forecast_recursive(history_df, horizon=0)

    def test_raises_when_target_missing(self, fitted_seasonal_ma):
        df = pd.DataFrame({"other": [1., 2., 3.]})
        with pytest.raises(ValueError, match="Target column"):
            fitted_seasonal_ma.forecast_recursive(df, horizon=2)

    def test_make_next_feature_row_is_ignored(self, fitted_seasonal_ma, seasonal_series):
        # MA 모델은 make_next_feature_row를 무시한다
        history_df = pd.DataFrame({"val": seasonal_series})
        result_none = fitted_seasonal_ma.forecast_recursive(history_df, horizon=3)
        result_with = fitted_seasonal_ma.forecast_recursive(
            history_df, horizon=3, make_next_feature_row=lambda df, step: None
        )
        assert result_none["predicted_val"].tolist() == pytest.approx(
            result_with["predicted_val"].tolist()
        )

    def test_timestamp_col_inferred(self, fitted_seasonal_ma, seasonal_series):
        ts = pd.date_range("2024-01-01", periods=len(seasonal_series), freq="h")
        history_df = pd.DataFrame({"val": seasonal_series, "ts": ts})
        result = fitted_seasonal_ma.forecast_recursive(history_df, horizon=3, timestamp_col="ts")
        assert "ts" in result.columns


class TestSaveLoadMA:
    def test_save_creates_file(self, fitted_seasonal_ma, tmp_path):
        path = fitted_seasonal_ma.save(tmp_path / "ma_model.joblib")
        assert path.exists()

    def test_save_raises_when_not_fitted(self, tmp_path):
        m = MovingAverageForecaster("val")
        with pytest.raises(RuntimeError, match="not fitted"):
            m.save(tmp_path / "model.joblib")

    def test_load_restores_metadata(self, fitted_seasonal_ma, tmp_path):
        path = fitted_seasonal_ma.save(tmp_path / "model.joblib")
        loaded = MovingAverageForecaster.load(path)
        assert loaded.kind == "seasonal"
        assert loaded.window == 2
        assert loaded.seasonal_period == 4

    def test_load_restores_is_fitted(self, fitted_seasonal_ma, tmp_path):
        path = fitted_seasonal_ma.save(tmp_path / "model.joblib")
        loaded = MovingAverageForecaster.load(path)
        assert loaded.is_fitted is True

    def test_load_can_predict_frame(self, fitted_seasonal_ma, tmp_path):
        path = fitted_seasonal_ma.save(tmp_path / "model.joblib")
        loaded = MovingAverageForecaster.load(path)
        result = loaded.predict_frame(horizon=2)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_load_residuals_preserved(self, fitted_seasonal_ma, tmp_path):
        path = fitted_seasonal_ma.save(tmp_path / "model.joblib")
        loaded = MovingAverageForecaster.load(path)
        assert loaded._residuals is not None
        assert len(loaded._residuals) > 0
