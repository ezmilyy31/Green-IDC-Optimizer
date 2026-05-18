"""
domain/forecasting/lgbm_model.py 단위 테스트.

대상: LGBMForecaster — 초기화, 학습, 추론, 재귀 예측, feature importance, 저장/로드.
"""
import numpy as np
import pandas as pd
import pytest

from domain.forecasting.lgbm_model import DEFAULT_LGBM_PARAMS, LGBMForecaster


class TestInit:
    def test_default_params_merged(self):
        forecaster = LGBMForecaster("y")
        assert forecaster.params["n_estimators"] == DEFAULT_LGBM_PARAMS["n_estimators"]

    def test_custom_params_override_defaults(self, fast_lgbm_params):
        forecaster = LGBMForecaster("y", params=fast_lgbm_params)
        assert forecaster.params["n_estimators"] == 5
        # 기본값은 그대로 유지
        assert forecaster.params["objective"] == "regression"

    def test_is_fitted_false_on_init(self):
        assert LGBMForecaster("y").is_fitted is False

    def test_feature_columns_stored(self):
        forecaster = LGBMForecaster("y", feature_columns=["a", "b"])
        assert forecaster.feature_columns == ["a", "b"]

    def test_empty_feature_columns_default(self):
        forecaster = LGBMForecaster("y")
        assert forecaster.feature_columns == []


class TestFit:
    def test_fit_with_train_df_sets_is_fitted(self, fitted_lgbm):
        assert fitted_lgbm.is_fitted is True

    def test_fit_returns_self(self, tiny_train_df, fast_lgbm_params):
        forecaster = LGBMForecaster("it_load_kw", ["feat_a", "feat_b"], params=fast_lgbm_params)
        result = forecaster.fit(train_df=tiny_train_df)
        assert result is forecaster

    def test_fit_with_X_y(self, tiny_train_df, fast_lgbm_params):
        X = tiny_train_df[["feat_a", "feat_b"]]
        y = tiny_train_df["it_load_kw"]
        forecaster = LGBMForecaster("it_load_kw", ["feat_a", "feat_b"], params=fast_lgbm_params)
        forecaster.fit(X=X, y=y)
        assert forecaster.is_fitted is True

    def test_fit_infers_feature_columns_from_X(self, tiny_train_df, fast_lgbm_params):
        # feature_columns 없이 X/y 경로로 학습 시 X 컬럼을 자동 저장
        X = tiny_train_df[["feat_a", "feat_b"]]
        y = tiny_train_df["it_load_kw"]
        forecaster = LGBMForecaster("it_load_kw", params=fast_lgbm_params)
        forecaster.fit(X=X, y=y)
        assert forecaster.feature_columns == ["feat_a", "feat_b"]

    def test_fit_raises_when_nothing_provided(self, fast_lgbm_params):
        forecaster = LGBMForecaster("y", ["a"], params=fast_lgbm_params)
        with pytest.raises(ValueError, match="Either train_df or both X and y"):
            forecaster.fit()

    def test_fit_raises_when_target_missing_from_train_df(self, tiny_train_df, fast_lgbm_params):
        # it_load_kw 컬럼이 없는 DataFrame 전달
        df_no_target = tiny_train_df[["feat_a", "feat_b"]]
        forecaster = LGBMForecaster("it_load_kw", ["feat_a", "feat_b"], params=fast_lgbm_params)
        with pytest.raises(ValueError, match="Target column"):
            forecaster.fit(train_df=df_no_target)

    def test_fit_raises_when_feature_column_missing(self, tiny_train_df, fast_lgbm_params):
        # 존재하지 않는 feature 컬럼 지정
        forecaster = LGBMForecaster(
            "it_load_kw", ["feat_a", "nonexistent"], params=fast_lgbm_params
        )
        with pytest.raises(ValueError, match="Missing feature columns"):
            forecaster.fit(train_df=tiny_train_df)

    def test_fit_raises_when_X_is_not_dataframe(self, fast_lgbm_params):
        forecaster = LGBMForecaster("y", params=fast_lgbm_params)
        with pytest.raises(TypeError, match="pandas DataFrame"):
            forecaster.fit(X=np.array([[1, 2], [3, 4]]), y=np.array([1, 2]))


class TestPredict:
    def test_predict_returns_ndarray(self, fitted_lgbm, tiny_train_df):
        X = tiny_train_df[["feat_a", "feat_b"]]
        result = fitted_lgbm.predict(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (20,)

    def test_predict_dtype_is_float(self, fitted_lgbm, tiny_train_df):
        X = tiny_train_df[["feat_a", "feat_b"]]
        result = fitted_lgbm.predict(X)
        assert result.dtype == float

    def test_predict_raises_when_not_fitted(self, fast_lgbm_params):
        forecaster = LGBMForecaster("it_load_kw", ["feat_a"], params=fast_lgbm_params)
        X = pd.DataFrame({"feat_a": [1., 2.]})
        with pytest.raises(RuntimeError, match="not fitted"):
            forecaster.predict(X)

    def test_predict_raises_when_feature_missing(self, fitted_lgbm):
        # feat_b 컬럼 누락
        X = pd.DataFrame({"feat_a": [0.5, 0.3]})
        with pytest.raises(ValueError, match="Missing feature columns"):
            fitted_lgbm.predict(X)

    def test_predict_raises_when_X_not_dataframe(self, fitted_lgbm):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            fitted_lgbm.predict(np.array([[1., 2.]]))


class TestPredictFrame:
    def test_returns_dataframe(self, fitted_lgbm, tiny_train_df):
        X = tiny_train_df[["feat_a", "feat_b"]]
        result = fitted_lgbm.predict_frame(X)
        assert isinstance(result, pd.DataFrame)

    def test_default_column_name(self, fitted_lgbm, tiny_train_df):
        X = tiny_train_df[["feat_a", "feat_b"]]
        result = fitted_lgbm.predict_frame(X)
        assert "predicted_it_load_kw" in result.columns

    def test_custom_prediction_col(self, fitted_lgbm, tiny_train_df):
        X = tiny_train_df[["feat_a", "feat_b"]]
        result = fitted_lgbm.predict_frame(X, prediction_col="yhat")
        assert "yhat" in result.columns
        assert "predicted_it_load_kw" not in result.columns

    def test_includes_timestamp_col(self, fitted_lgbm, tiny_train_df):
        X = tiny_train_df[["feat_a", "feat_b"]].copy()
        X["ts"] = pd.date_range("2024-01-01", periods=20, freq="h")
        result = fitted_lgbm.predict_frame(X, timestamp_col="ts")
        assert "ts" in result.columns
        # timestamp 컬럼은 맨 앞에 위치
        assert result.columns[0] == "ts"

    def test_omits_missing_timestamp_col(self, fitted_lgbm, tiny_train_df):
        X = tiny_train_df[["feat_a", "feat_b"]]
        result = fitted_lgbm.predict_frame(X, timestamp_col="nonexistent")
        assert "nonexistent" not in result.columns


class TestForecastRecursiveLGBM:
    @pytest.fixture
    def history_df(self, tiny_train_df):
        return tiny_train_df.head(5)

    @pytest.fixture
    def make_next(self):
        return lambda df, step: pd.DataFrame({"feat_a": [1.], "feat_b": [2.]})

    def test_returns_dataframe_with_correct_len(self, fitted_lgbm, history_df, make_next):
        result = fitted_lgbm.forecast_recursive(
            history_df, horizon=3, make_next_feature_row=make_next
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_step_column_sequential(self, fitted_lgbm, history_df, make_next):
        result = fitted_lgbm.forecast_recursive(
            history_df, horizon=3, make_next_feature_row=make_next
        )
        assert result["step"].tolist() == [1, 2, 3]

    def test_contains_predicted_column(self, fitted_lgbm, history_df, make_next):
        result = fitted_lgbm.forecast_recursive(
            history_df, horizon=3, make_next_feature_row=make_next
        )
        assert "predicted_it_load_kw" in result.columns

    def test_raises_for_zero_horizon(self, fitted_lgbm, history_df, make_next):
        with pytest.raises(ValueError, match="greater than 0"):
            fitted_lgbm.forecast_recursive(
                history_df, horizon=0, make_next_feature_row=make_next
            )

    def test_raises_when_callback_returns_multi_row(self, fitted_lgbm, history_df):
        bad_callback = lambda df, step: pd.DataFrame(
            {"feat_a": [1., 2.], "feat_b": [2., 3.]}
        )
        with pytest.raises(ValueError, match="exactly one row"):
            fitted_lgbm.forecast_recursive(
                history_df, horizon=1, make_next_feature_row=bad_callback
            )

    def test_raises_when_callback_returns_non_dataframe(self, fitted_lgbm, history_df):
        bad_callback = lambda df, step: "not a dataframe"
        with pytest.raises(TypeError, match="pandas DataFrame or Series"):
            fitted_lgbm.forecast_recursive(
                history_df, horizon=1, make_next_feature_row=bad_callback
            )


class TestGetFeatureImportance:
    def test_returns_dataframe_with_correct_columns(self, fitted_lgbm):
        result = fitted_lgbm.get_feature_importance()
        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "importance" in result.columns

    def test_feature_count_matches(self, fitted_lgbm):
        result = fitted_lgbm.get_feature_importance()
        assert len(result) == 2  # feat_a, feat_b

    def test_normalize_sums_to_one(self, fitted_lgbm):
        result = fitted_lgbm.get_feature_importance(normalize=True)
        assert result["importance"].sum() == pytest.approx(1.0)

    def test_sorted_descending(self, fitted_lgbm):
        result = fitted_lgbm.get_feature_importance()
        importances = result["importance"].tolist()
        assert importances == sorted(importances, reverse=True)

    def test_raises_when_not_fitted(self, fast_lgbm_params):
        forecaster = LGBMForecaster("y", ["a"], params=fast_lgbm_params)
        with pytest.raises(RuntimeError, match="not fitted"):
            forecaster.get_feature_importance()


class TestSaveLoadLGBM:
    def test_save_returns_path(self, fitted_lgbm, tmp_path):
        path = fitted_lgbm.save(tmp_path / "model.joblib")
        assert path.exists()
        assert path.suffix == ".joblib"

    def test_save_raises_when_not_fitted(self, fast_lgbm_params, tmp_path):
        forecaster = LGBMForecaster("y", ["a"], params=fast_lgbm_params)
        with pytest.raises(RuntimeError, match="not fitted"):
            forecaster.save(tmp_path / "model.joblib")

    def test_load_restores_is_fitted(self, fitted_lgbm, tmp_path):
        path = fitted_lgbm.save(tmp_path / "model.joblib")
        loaded = LGBMForecaster.load(path)
        assert loaded.is_fitted is True

    def test_load_restores_target_name(self, fitted_lgbm, tmp_path):
        path = fitted_lgbm.save(tmp_path / "model.joblib")
        loaded = LGBMForecaster.load(path)
        assert loaded.target_name == "it_load_kw"

    def test_load_restores_feature_columns(self, fitted_lgbm, tmp_path):
        path = fitted_lgbm.save(tmp_path / "model.joblib")
        loaded = LGBMForecaster.load(path)
        assert loaded.feature_columns == ["feat_a", "feat_b"]

    def test_load_can_predict_after_reload(self, fitted_lgbm, tiny_train_df, tmp_path):
        path = fitted_lgbm.save(tmp_path / "model.joblib")
        loaded = LGBMForecaster.load(path)
        X = tiny_train_df[["feat_a", "feat_b"]]
        result = loaded.predict(X)
        assert result.shape == (20,)
