"""PIDController 단위 테스트."""

import pytest

from domain.controllers.pid import (
    PIDController,
    SUPPLY_TEMP_MAX_C,
    SUPPLY_TEMP_MIN_C,
)


class TestInitialState:
    def test_initial_supply_is_output_max(self):
        pid = PIDController()
        assert pid._supply == SUPPLY_TEMP_MAX_C
        assert pid._prev_error == 0.0

    def test_default_setpoint_is_24(self):
        pid = PIDController()
        assert pid.setpoint == 24.0


class TestSetpointTracking:
    def test_supply_decreases_when_zone_above_setpoint(self):
        # zone 26°C, setpoint 24°C → 너무 더움 → supply 내려야 함
        pid = PIDController(setpoint=24.0)
        prev = pid._supply
        out = pid.compute(current_value=26.0, dt=1.0)
        assert out < prev

    def test_supply_increases_when_zone_below_setpoint(self):
        pid = PIDController(setpoint=24.0)
        # 한 번 내려서 _supply 가 max 가 아닌 상태로 만든 후 비교
        pid.compute(current_value=26.0, dt=1.0)
        prev = pid._supply
        out = pid.compute(current_value=22.0, dt=1.0)
        assert out > prev


class TestAntiWindup:
    def test_output_clamped_to_min_under_huge_positive_error(self):
        # current=100 → error=24-100=-76 (zone 너무 더움)
        # 적분항 포함 시 new_supply 가 output_min 한참 아래 → anti-windup 분기 진입
        # → 결과는 output_min 으로 clamp
        pid = PIDController()
        out = pid.compute(current_value=100.0, dt=1.0)
        assert out == pytest.approx(SUPPLY_TEMP_MIN_C)
        assert pid._supply == pytest.approx(SUPPLY_TEMP_MIN_C)

    def test_output_clamped_to_max_under_huge_negative_error(self):
        # current=-50 → error=24-(-50)=+74 → delta_supply 양의 큰 값 → output_max 초과
        pid = PIDController()
        out = pid.compute(current_value=-50.0, dt=1.0)
        assert out == pytest.approx(SUPPLY_TEMP_MAX_C)

    def test_prev_error_updated_even_on_saturation(self):
        # anti-windup 분기에서도 _prev_error 는 갱신되어야 함
        pid = PIDController(setpoint=24.0)
        pid.compute(current_value=100.0, dt=1.0)
        assert pid._prev_error == pytest.approx(24.0 - 100.0)

    def test_supply_recovers_from_saturation(self):
        # 1) saturation 발생 → supply 가 min 에 묶임
        # 2) 오차 반전(현재값이 setpoint 아래) → supply 가 min 에서 벗어나야 함
        pid = PIDController(setpoint=24.0)
        for _ in range(5):
            pid.compute(current_value=100.0, dt=1.0)
        assert pid._supply == pytest.approx(SUPPLY_TEMP_MIN_C)

        # 오차 반전: 매우 차가운 zone 으로 변경
        pid.compute(current_value=10.0, dt=1.0)
        assert pid._supply > SUPPLY_TEMP_MIN_C  # 더 이상 클램프 아님


class TestOutputBounds:
    @pytest.mark.parametrize("zone_temp", [-100.0, -10.0, 0.0, 24.0, 50.0, 100.0])
    def test_output_always_within_bounds(self, zone_temp):
        pid = PIDController()
        out = pid.compute(current_value=zone_temp, dt=1.0)
        assert SUPPLY_TEMP_MIN_C <= out <= SUPPLY_TEMP_MAX_C


class TestReset:
    def test_reset_restores_initial_state(self):
        pid = PIDController()
        # 상태 변경
        pid.compute(current_value=26.0, dt=1.0)
        pid.compute(current_value=28.0, dt=1.0)
        assert pid._prev_error != 0.0
        # reset
        pid.reset()
        assert pid._prev_error == 0.0
        assert pid._supply == SUPPLY_TEMP_MAX_C
