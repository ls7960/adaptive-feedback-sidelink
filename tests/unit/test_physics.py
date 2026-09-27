import math
import sys

import pytest

from adaptive_feedback_llm.models import TrafficClass
from adaptive_feedback_llm.physics import (
    bottleneck_rate_bps,
    classified_margin,
    db_to_linear,
    delay_margin,
    flow_utility,
    link_loss_probability,
    link_success_probability,
    path_reliability,
    pipelined_completion_delay_s,
    rate_per_sru_bps,
    reliability_margin,
)


def test_db_to_linear_and_single_multi_sru_rates() -> None:
    assert db_to_linear(0.0) == pytest.approx(1.0)
    assert db_to_linear(10.0) == pytest.approx(10.0)
    one_sru = rate_per_sru_bps(10.0, 180000.0)

    assert one_sru == pytest.approx(180000.0 * math.log2(11.0))
    assert 3 * one_sru == pytest.approx(3 * 180000.0 * math.log2(11.0))


def test_db_to_linear_rejects_negative_underflow_to_zero() -> None:
    with pytest.raises(OverflowError, match="underflows to zero"):
        db_to_linear(-4000.0)


def test_bottleneck_rate_and_pipelined_completion_delay() -> None:
    bottleneck = bottleneck_rate_bps((4.0, 10.0, 6.0))
    delay = pipelined_completion_delay_s(100.0, bottleneck, (0.1, 0.2, 0.3))

    assert bottleneck == 4.0
    assert delay == pytest.approx(25.6)
    assert delay != pytest.approx(0.6 + 100.0 / 4.0 + 100.0 / 10.0 + 100.0 / 6.0)


def test_frozen_loss_calibration_and_success() -> None:
    at_operating = link_loss_probability(0.0, 0.0)
    four_db_above = link_loss_probability(4.0, 0.0)

    assert at_operating == pytest.approx(0.1)
    assert four_db_above == pytest.approx(0.01)
    assert link_success_probability(4.0, 0.0) == pytest.approx(1.0 - four_db_above)


def test_path_reliability_is_product_and_independent_of_sru_count() -> None:
    per_hop = (link_success_probability(4.0, 0.0), link_success_probability(10.0, 0.0))
    reliability_with_one_sru = path_reliability(per_hop)
    reliability_with_many_srus = path_reliability(per_hop)

    assert reliability_with_one_sru == pytest.approx(per_hop[0] * per_hop[1])
    assert reliability_with_many_srus == reliability_with_one_sru


def test_margin_and_utility_endpoints() -> None:
    assert delay_margin(10.0, 10.0, 1.0e-9) == 0.0
    assert delay_margin(0.0, 10.0, 1.0e-9) == 1.0
    assert reliability_margin(0.8, 0.8, 1.0e-9) == 0.0
    assert reliability_margin(1.0, 0.8, 1.0e-9) == 1.0
    assert flow_utility(3.0, 0.0) == 3.0
    assert flow_utility(3.0, 1.0) == 6.0


def test_flow_utility_rejects_nonfinite_product() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        flow_utility(sys.float_info.max, 1.0)


def test_classification_selects_only_its_own_post_qos_margin() -> None:
    common = dict(
        completion_delay_s=5.0,
        deadline_s=10.0,
        reliability=0.9,
        minimum_reliability=0.8,
        tolerance=1.0e-9,
    )
    assert classified_margin(TrafficClass.DELAY, **common) == pytest.approx(0.5)
    assert classified_margin(TrafficClass.RELIABILITY, **common) == pytest.approx(0.5)


def test_float_tolerance_clips_only_micro_boundary_excursions() -> None:
    tolerance = 1.0e-9
    assert delay_margin(1.0 + tolerance / 2.0, 1.0, tolerance) == 0.0
    assert reliability_margin(0.8 - tolerance / 2.0, 0.8, tolerance) == 0.0

    with pytest.raises(ValueError):
        delay_margin(1.0 + 2.0 * tolerance, 1.0, tolerance)
    with pytest.raises(ValueError):
        reliability_margin(0.8 - 2.0 * tolerance, 0.8, tolerance)


@pytest.mark.parametrize(
    ("function", "args", "expected_error"),
    [
        (db_to_linear, (True,), TypeError),
        (db_to_linear, (float("nan"),), ValueError),
        (rate_per_sru_bps, (0.0, 0.0), ValueError),
        (bottleneck_rate_bps, ((1.0, 0.0),), ValueError),
        (path_reliability, ((1.1,),), ValueError),
        (flow_utility, (0.0, 0.5), ValueError),
    ],
)
def test_physics_rejects_invalid_numbers(function, args, expected_error) -> None:
    with pytest.raises(expected_error):
        function(*args)
