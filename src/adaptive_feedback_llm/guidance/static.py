"""Nonbinding guidance derived only from static single-flow information."""

from __future__ import annotations

import math
from collections import Counter

from pydantic import Field, model_validator

from adaptive_feedback_llm.models import (
    CandidatePath,
    EvaluatorScenario,
    SRUAllocation,
    TrafficClass,
)
from adaptive_feedback_llm.models.domain import FrozenStrictModel
from adaptive_feedback_llm.physics import delay_margin, reliability_margin
from adaptive_feedback_llm.solvers.common import construct_path_allocation


class PathGuidance(FrozenStrictModel):
    """Static path attributes and nonbinding rank."""

    flow_id: int = Field(ge=0)
    path_id: int = Field(ge=0)
    rank: int = Field(ge=0)
    single_flow_feasible: bool
    required_sru_total: int | None = Field(default=None, gt=0)
    reference_sru_allocations: tuple[SRUAllocation, ...] = ()
    hotspot_burden: float = Field(ge=0.0, allow_inf_nan=False)
    class_quality_rank: int | None = Field(default=None, ge=0)
    hop_count: int = Field(gt=0)
    node_sequence: tuple[int, ...]

    @model_validator(mode="after")
    def validate_reference_allocation(self) -> "PathGuidance":
        expected_edges = tuple(zip(self.node_sequence, self.node_sequence[1:]))
        actual_edges = tuple(item.edge for item in self.reference_sru_allocations)
        if self.single_flow_feasible:
            if self.required_sru_total is None or actual_edges != expected_edges:
                raise ValueError(
                    "feasible guidance requires one ordered reference allocation per path edge"
                )
            if sum(item.num_srus for item in self.reference_sru_allocations) != self.required_sru_total:
                raise ValueError("reference allocation must sum to required_sru_total")
            if any(item.num_srus <= 0 for item in self.reference_sru_allocations):
                raise ValueError("reference allocation SRUs must be positive")
        elif self.required_sru_total is not None or self.reference_sru_allocations:
            raise ValueError(
                "infeasible guidance cannot expose a reference allocation or required total"
            )
        return self


class StaticGuidance(FrozenStrictModel):
    """All static path guidance records, ordered by flow then rank."""

    paths: tuple[PathGuidance, ...]

    def rank_for(self, flow_id: int, path_id: int) -> int:
        """Return the frozen rank for one catalog identity."""

        for item in self.paths:
            if item.flow_id == flow_id and item.path_id == path_id:
                return item.rank
        raise KeyError((flow_id, path_id))


def build_static_guidance(scenario: EvaluatorScenario) -> StaticGuidance:
    """Rank paths without observing a selected schedule or residual capacity."""

    occurrence_counts = Counter(
        edge
        for catalog in scenario.path_catalogs
        for path in catalog.paths
        for edge in path.edges
    )
    links = scenario.link_by_edge()
    full_capacity = {edge: link.capacity_srus for edge, link in links.items()}
    flows = scenario.flow_by_id()
    records: list[PathGuidance] = []
    for catalog in scenario.path_catalogs:
        flow = flows[catalog.flow_id]
        candidates: list[
            tuple[
                tuple[object, ...],
                CandidatePath,
                int | None,
                float | None,
                float,
                tuple[SRUAllocation, ...],
            ]
        ] = []
        for path in catalog.paths:
            construction = construct_path_allocation(
                scenario, flow, path, full_capacity
            )
            hotspot = math.fsum(
                occurrence_counts[edge] / links[edge].capacity_srus
                for edge in path.edges
            )
            if construction is None:
                required_total = None
                quality = None
                reference_allocations: tuple[SRUAllocation, ...] = ()
                key: tuple[object, ...] = (
                    True,
                    math.inf,
                    hotspot,
                    math.inf,
                    path.hop_count,
                    path.node_sequence,
                )
            else:
                reference_allocations = construction.allocations
                required_total = sum(
                    allocation.num_srus for allocation in reference_allocations
                )
                if flow.traffic_class is TrafficClass.DELAY:
                    quality = delay_margin(
                        construction.completion_delay_s,
                        flow.deadline_s,
                        scenario.numerics.float_tolerance,
                    )
                else:
                    quality = reliability_margin(
                        construction.reliability,
                        flow.min_reliability,
                        scenario.numerics.float_tolerance,
                    )
                key = (
                    False,
                    required_total,
                    hotspot,
                    -quality,
                    path.hop_count,
                    path.node_sequence,
                )
            candidates.append(
                (
                    key,
                    path,
                    required_total,
                    quality,
                    hotspot,
                    reference_allocations,
                )
            )

        candidates.sort(key=lambda item: item[0])
        quality_order = sorted(
            (-quality, path.path_id)
            for _, path, _, quality, _, _ in candidates
            if quality is not None
        )
        quality_rank = {
            path_id: rank for rank, (_, path_id) in enumerate(quality_order)
        }
        for rank, (
            _,
            path,
            required_total,
            quality,
            hotspot,
            reference_allocations,
        ) in enumerate(candidates):
            records.append(
                PathGuidance(
                    flow_id=flow.flow_id,
                    path_id=path.path_id,
                    rank=rank,
                    single_flow_feasible=required_total is not None,
                    required_sru_total=required_total,
                    reference_sru_allocations=reference_allocations,
                    hotspot_burden=hotspot,
                    class_quality_rank=(
                        quality_rank[path.path_id] if quality is not None else None
                    ),
                    hop_count=path.hop_count,
                    node_sequence=path.node_sequence,
                )
            )
    return StaticGuidance(paths=tuple(records))
