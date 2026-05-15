"""Rule-based 컨트롤러 테스트 (wet-bulb 임계값 기반).

임계값: free < 10°C ≤ hybrid < 18°C ≤ chiller
습구온도 근사식: T_wb ≈ T - (1 - RH/100) × (T - 4)
50% RH: T_wb = 0.5T + 2
"""

import pytest

from core.config.enums import CoolingMode
from domain.controllers.rule_based import (
    calculate_setpoint,
    decide_cooling_mode,
    run_rule_based,
)


class TestDecideCoolingMode:
    def test_low_wet_bulb_is_free_cooling(self):
        # T=10°C, RH=50% → wb=7°C < 10 → FREE
        assert decide_cooling_mode(10.0, 50.0) == CoolingMode.FREE_COOLING
        # T=15°C, RH=50% → wb=9.5°C < 10 → FREE
        assert decide_cooling_mode(15.0, 50.0) == CoolingMode.FREE_COOLING

    def test_free_cooling_boundary(self):
        # T=16°C, RH=50% → wb=10°C → HYBRID (조건: wb < 10 → FREE)
        assert decide_cooling_mode(16.0, 50.0) == CoolingMode.HYBRID

    def test_hybrid_range(self):
        # T=20°C, RH=50% → wb=12°C → HYBRID
        assert decide_cooling_mode(20.0, 50.0) == CoolingMode.HYBRID
        # T=30°C, RH=50% → wb=17°C < 18 → HYBRID
        assert decide_cooling_mode(30.0, 50.0) == CoolingMode.HYBRID

    def test_hybrid_upper_boundary(self):
        # T=32°C, RH=50% → wb=18°C → CHILLER (조건: wb < 18 → HYBRID)
        assert decide_cooling_mode(32.0, 50.0) == CoolingMode.CHILLER

    def test_humid_summer_is_chiller(self):
        # T=30°C, RH=90% → wb≈27.4°C → CHILLER (잠열 부하 큼)
        assert decide_cooling_mode(30.0, 90.0) == CoolingMode.CHILLER

    def test_dry_climate_extended_free_cooling(self):
        # T=18°C, RH=20% → wb≈6.8°C → FREE (건조하면 자유공조 가능)
        assert decide_cooling_mode(18.0, 20.0) == CoolingMode.FREE_COOLING


class TestCalculateSetpoint:
    def test_free_cooling_setpoint(self):
        assert calculate_setpoint(CoolingMode.FREE_COOLING, 10.0) == 22.0

    def test_hybrid_setpoint(self):
        assert calculate_setpoint(CoolingMode.HYBRID, 20.0) == 20.0

    def test_chiller_setpoint(self):
        assert calculate_setpoint(CoolingMode.CHILLER, 30.0) == 18.0


class TestRunRuleBased:
    def test_free_cooling_full_ratio(self):
        # T=10°C, RH=50% → wb=7°C → FREE
        result = run_rule_based(outdoor_temp_c=10.0, outdoor_humidity=50.0, it_power_kw=200.0)
        assert result.cooling_mode == CoolingMode.FREE_COOLING
        assert result.supply_air_temp_setpoint_c == 22.0
        assert result.free_cooling_ratio == 1.0

    def test_chiller_zero_ratio(self):
        # 습한 여름: T=30°C, RH=90% → wb≈27.4 → CHILLER
        result = run_rule_based(outdoor_temp_c=30.0, outdoor_humidity=90.0, it_power_kw=200.0)
        assert result.cooling_mode == CoolingMode.CHILLER
        assert result.supply_air_temp_setpoint_c == 18.0
        assert result.free_cooling_ratio == 0.0

    def test_hybrid_lower_boundary_ratio_one(self):
        # T=16°C, RH=50% → wb=10.0 → HYBRID 시작점 → ratio ≈ 1.0
        result = run_rule_based(outdoor_temp_c=16.0, outdoor_humidity=50.0, it_power_kw=200.0)
        assert result.cooling_mode == CoolingMode.HYBRID
        assert result.free_cooling_ratio == pytest.approx(1.0)

    def test_hybrid_upper_boundary_ratio_zero(self):
        # T=31.999°C, RH=50% → wb≈18.0 (직전) → HYBRID 끝점 → ratio ≈ 0.0
        result = run_rule_based(outdoor_temp_c=31.99, outdoor_humidity=50.0, it_power_kw=200.0)
        assert result.cooling_mode == CoolingMode.HYBRID
        assert result.free_cooling_ratio == pytest.approx(0.0, abs=0.01)

    def test_hybrid_midpoint_ratio_half(self):
        # T=24°C, RH=50% → wb=14.0 → 중간 → ratio = 1 - (14-10)/8 = 0.5
        result = run_rule_based(outdoor_temp_c=24.0, outdoor_humidity=50.0, it_power_kw=200.0)
        assert result.cooling_mode == CoolingMode.HYBRID
        assert result.free_cooling_ratio == pytest.approx(0.5)

    def test_hybrid_ratio_monotonic_decreasing(self):
        # 같은 습도에서 외기 ↑ → wet-bulb ↑ → ratio ↓
        ratios = [
            run_rule_based(t, 50.0, 200.0).free_cooling_ratio
            for t in [16.0, 20.0, 24.0, 28.0, 31.0]
        ]
        assert ratios == sorted(ratios, reverse=True)
