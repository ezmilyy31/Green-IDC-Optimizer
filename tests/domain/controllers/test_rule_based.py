import pytest

from core.config.enums import CoolingMode
from domain.controllers.rule_based import (
    calculate_setpoint,
    decide_cooling_mode,
    run_rule_based,
)


class TestDecideCoolingMode:
    def test_below_threshold_is_free_cooling(self):
        assert decide_cooling_mode(10.0) == CoolingMode.FREE_COOLING
        assert decide_cooling_mode(14.99) == CoolingMode.FREE_COOLING

    def test_lower_boundary_is_hybrid(self):
        # 15.0 정확히는 HYBRID (조건: outdoor < 15 → FREE)
        assert decide_cooling_mode(15.0) == CoolingMode.HYBRID

    def test_mid_range_is_hybrid(self):
        assert decide_cooling_mode(18.5) == CoolingMode.HYBRID

    def test_upper_boundary_is_hybrid(self):
        # 22.0 정확히는 HYBRID (조건: outdoor <= 22 → HYBRID)
        assert decide_cooling_mode(22.0) == CoolingMode.HYBRID

    def test_above_threshold_is_chiller(self):
        assert decide_cooling_mode(22.01) == CoolingMode.CHILLER
        assert decide_cooling_mode(30.0) == CoolingMode.CHILLER


class TestCalculateSetpoint:
    def test_free_cooling_setpoint(self):
        assert calculate_setpoint(CoolingMode.FREE_COOLING, 10.0) == 22.0

    def test_hybrid_setpoint(self):
        assert calculate_setpoint(CoolingMode.HYBRID, 18.0) == 20.0

    def test_chiller_setpoint(self):
        assert calculate_setpoint(CoolingMode.CHILLER, 30.0) == 18.0


class TestRunRuleBased:
    def test_free_cooling_full_ratio(self):
        result = run_rule_based(outdoor_temp_c=10.0, outdoor_humidity=50.0, it_power_kw=200.0)
        assert result.cooling_mode == CoolingMode.FREE_COOLING
        assert result.supply_air_temp_setpoint_c == 22.0
        assert result.free_cooling_ratio == 1.0

    def test_chiller_zero_ratio(self):
        result = run_rule_based(outdoor_temp_c=30.0, outdoor_humidity=50.0, it_power_kw=200.0)
        assert result.cooling_mode == CoolingMode.CHILLER
        assert result.supply_air_temp_setpoint_c == 18.0
        assert result.free_cooling_ratio == 0.0

    def test_hybrid_lower_boundary_ratio_one(self):
        # T=15 → 1 - (15-15)/(22-15) = 1.0
        result = run_rule_based(outdoor_temp_c=15.0, outdoor_humidity=50.0, it_power_kw=200.0)
        assert result.cooling_mode == CoolingMode.HYBRID
        assert result.free_cooling_ratio == pytest.approx(1.0)

    def test_hybrid_upper_boundary_ratio_zero(self):
        # T=22 → 1 - (22-15)/(22-15) = 0.0
        result = run_rule_based(outdoor_temp_c=22.0, outdoor_humidity=50.0, it_power_kw=200.0)
        assert result.cooling_mode == CoolingMode.HYBRID
        assert result.free_cooling_ratio == pytest.approx(0.0)

    def test_hybrid_midpoint_ratio_half(self):
        # T=18.5 → 1 - 3.5/7 = 0.5
        result = run_rule_based(outdoor_temp_c=18.5, outdoor_humidity=50.0, it_power_kw=200.0)
        assert result.cooling_mode == CoolingMode.HYBRID
        assert result.free_cooling_ratio == pytest.approx(0.5)

    def test_hybrid_ratio_monotonic_decreasing(self):
        ratios = [
            run_rule_based(t, 50.0, 200.0).free_cooling_ratio
            for t in [15.0, 17.0, 19.0, 21.0, 22.0]
        ]
        assert ratios == sorted(ratios, reverse=True)
