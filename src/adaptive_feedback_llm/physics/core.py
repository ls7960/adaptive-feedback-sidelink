"""Centralized deterministic physical and utility formulas."""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable

from adaptive_feedback_llm.config import RELIABILITY_SLOPE_PER_DB
from adaptive_feedback_llm.models.domain import TrafficClass

_LN_2 = math.log(2.0)
_LN_9 = math.log(9.0)
_LN_10_OVER_10 = math.log(10.0) / 10.0
_LOG_MAX_FLOAT = math.log(sys.float_info.max)


def _finite_number(name: str, value: float | int) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a real number, not {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def db_to_linear(sinr_db: float) -> float:
    """Convert finite SINR dB to linear form with explicit overflow handling."""

    value = _finite_number("sinr_db", sinr_db)
    exponent = value * _LN_10_OVER_10
    if exponent > _LOG_MAX_FLOAT:
        raise OverflowError("linear SINR is not representable as a finite float")
    result = math.exp(exponent)
    if result == 0.0:
        raise OverflowError("linear SINR underflows to zero and is not representable")
    return result


def rate_per_sru_bps(sinr_db: float, sru_bandwidth_hz: float) -> float:
    """Compute `W_SRU * log2(1 + gamma)` without exponential overflow."""

    value_db = _finite_number("sinr_db", sinr_db)
    bandwidth = _finite_number("sru_bandwidth_hz", sru_bandwidth_hz)
    if bandwidth <= 0.0:
        raise ValueError("sru_bandwidth_hz must be positive")
    log_gamma = value_db * _LN_10_OVER_10
    if log_gamma >= 0.0:
        log_one_plus_gamma = log_gamma + math.log1p(math.exp(-log_gamma))
    else:
        log_one_plus_gamma = math.log1p(math.exp(log_gamma))
    rate = bandwidth * log_one_plus_gamma / _LN_2
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("per-SRU rate must be finite and positive")
    return rate


def link_loss_probability(sinr_db: float, operating_sinr_db: float) -> float:
    """Compute the frozen exact-slope logistic link loss probability."""

    value_db = _finite_number("sinr_db", sinr_db)
    operating_db = _finite_number("operating_sinr_db", operating_sinr_db)
    z = RELIABILITY_SLOPE_PER_DB * (value_db - operating_db) + _LN_9
    if z >= 0.0:
        exp_negative = math.exp(-z)
        return exp_negative / (1.0 + exp_negative)
    exp_positive = math.exp(z)
    return 1.0 / (1.0 + exp_positive)


def link_success_probability(sinr_db: float, operating_sinr_db: float) -> float:
    """Return one minus the centralized link loss probability."""

    return 1.0 - link_loss_probability(sinr_db, operating_sinr_db)


def path_reliability(success_probabilities: Iterable[float]) -> float:
    """Multiply independent per-hop effective delivery probabilities."""

    probabilities = tuple(success_probabilities)
    if not probabilities:
        raise ValueError("path reliability requires at least one hop")
    product = 1.0
    for probability in probabilities:
        value = _finite_number("success_probability", probability)
        if not 0.0 <= value <= 1.0:
            raise ValueError("success probabilities must lie in [0, 1]")
        product *= value
    return product


def bottleneck_rate_bps(hop_rates_bps: Iterable[float]) -> float:
    """Return the minimum positive finite allocated hop rate."""

    rates = tuple(_finite_number("hop_rate_bps", rate) for rate in hop_rates_bps)
    if not rates:
        raise ValueError("bottleneck rate requires at least one hop")
    if any(rate <= 0.0 for rate in rates):
        raise ValueError("hop rates must be positive")
    return min(rates)


def pipelined_completion_delay_s(
    data_bits: float,
    bottleneck_rate: float,
    fixed_delays_s: Iterable[float],
) -> float:
    """Compute `sum(delta_e) + Q/R` under the frozen fluid-pipeline model."""

    demand = _finite_number("data_bits", data_bits)
    rate = _finite_number("bottleneck_rate_bps", bottleneck_rate)
    delays = tuple(_finite_number("fixed_delay_s", delay) for delay in fixed_delays_s)
    if demand <= 0.0:
        raise ValueError("data_bits must be positive")
    if rate <= 0.0:
        raise ValueError("bottleneck_rate_bps must be positive")
    if not delays or any(delay <= 0.0 for delay in delays):
        raise ValueError("a path requires positive fixed delay on every hop")
    result = math.fsum(delays) + demand / rate
    if not math.isfinite(result):
        raise ValueError("completion delay must be finite")
    return result


def delay_margin(delay_s: float, deadline_s: float, tolerance: float) -> float:
    """Return normalized delay margin after a tolerance-aware QoS check."""

    delay = _finite_number("delay_s", delay_s)
    deadline = _finite_number("deadline_s", deadline_s)
    tol = _finite_number("tolerance", tolerance)
    if delay < 0.0 or deadline <= 0.0 or tol <= 0.0:
        raise ValueError("delay must be non-negative; deadline and tolerance positive")
    if delay > deadline + tol:
        raise ValueError("delay violates its QoS deadline")
    raw_margin = 1.0 - delay / deadline
    if raw_margin < 0.0 and delay - deadline <= tol:
        return 0.0
    if raw_margin > 1.0 and raw_margin - 1.0 <= tol:
        return 1.0
    if not 0.0 <= raw_margin <= 1.0:
        raise ValueError("delay margin is outside [0, 1]")
    return raw_margin


def reliability_margin(
    reliability: float,
    minimum_reliability: float,
    tolerance: float,
) -> float:
    """Return normalized reliability gain after a tolerance-aware QoS check."""

    actual = _finite_number("reliability", reliability)
    minimum = _finite_number("minimum_reliability", minimum_reliability)
    tol = _finite_number("tolerance", tolerance)
    if not 0.0 <= minimum < 1.0 or tol <= 0.0:
        raise ValueError("minimum reliability must be in [0, 1); tolerance positive")
    if actual < -tol or actual > 1.0 + tol:
        raise ValueError("reliability is outside [0, 1]")
    if actual + tol < minimum:
        raise ValueError("reliability violates its QoS threshold")
    clipped = min(1.0, max(0.0, actual))
    raw_margin = (clipped - minimum) / (1.0 - minimum)
    if raw_margin < 0.0 and minimum - actual <= tol:
        return 0.0
    if raw_margin > 1.0 and actual - 1.0 <= tol:
        return 1.0
    if not 0.0 <= raw_margin <= 1.0:
        raise ValueError("reliability margin is outside [0, 1]")
    return raw_margin


def classified_margin(
    traffic_class: TrafficClass,
    *,
    completion_delay_s: float,
    deadline_s: float,
    reliability: float,
    minimum_reliability: float,
    tolerance: float,
) -> float:
    """Select exactly one post-QoS margin according to traffic class."""

    if traffic_class is TrafficClass.DELAY:
        return delay_margin(completion_delay_s, deadline_s, tolerance)
    if traffic_class is TrafficClass.RELIABILITY:
        return reliability_margin(reliability, minimum_reliability, tolerance)
    raise ValueError(f"unknown traffic class: {traffic_class!r}")


def flow_utility(priority: float, margin: float) -> float:
    """Compute the unscaled served-flow utility `rho * (1 + margin)`."""

    priority_value = _finite_number("priority", priority)
    margin_value = _finite_number("margin", margin)
    if priority_value <= 0.0:
        raise ValueError("priority must be positive")
    if not 0.0 <= margin_value <= 1.0:
        raise ValueError("margin must lie in [0, 1]")
    result = priority_value * (1.0 + margin_value)
    if not math.isfinite(result):
        raise ValueError("flow utility must be finite")
    return result
