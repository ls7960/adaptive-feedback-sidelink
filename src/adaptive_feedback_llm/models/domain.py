"""Strict immutable domain models for deterministic scenarios."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adaptive_feedback_llm.config import NumericsConfig, PhysicsConfig


class FrozenStrictModel(BaseModel):
    """Immutable Pydantic base that rejects unknown fields and coercion."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TrafficClass(str, Enum):
    """Traffic class selects the rewarded margin after both QoS checks pass."""

    DELAY = "delay"
    RELIABILITY = "reliability"


class DirectedLink(FrozenStrictModel):
    """One directed link with evaluator-private physical state."""

    src: int = Field(ge=0)
    dst: int = Field(ge=0)
    capacity_srus: int = Field(gt=0)
    sinr_db: float = Field(allow_inf_nan=False)
    fixed_delay_s: float = Field(gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def reject_self_loop(self) -> "DirectedLink":
        if self.src == self.dst:
            raise ValueError("directed links cannot be self-loops")
        return self

    @property
    def edge(self) -> tuple[int, int]:
        """Return the directed edge identity."""

        return (self.src, self.dst)


class Flow(FrozenStrictModel):
    """One traffic demand with both mandatory QoS thresholds."""

    flow_id: int = Field(ge=0)
    source: int = Field(ge=0)
    target: int = Field(ge=0)
    data_bits: float = Field(gt=0.0, allow_inf_nan=False)
    priority: float = Field(gt=0.0, allow_inf_nan=False)
    traffic_class: TrafficClass
    deadline_s: float = Field(gt=0.0, allow_inf_nan=False)
    min_reliability: float = Field(ge=0.0, lt=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def reject_same_endpoint(self) -> "Flow":
        if self.source == self.target:
            raise ValueError("flow source and target must differ")
        return self


class CandidatePath(FrozenStrictModel):
    """Stable path-catalog member; paths are supplied by fixtures."""

    flow_id: int = Field(ge=0)
    path_id: int = Field(ge=0)
    node_sequence: tuple[int, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_simple_path(self) -> "CandidatePath":
        if any(node_id < 0 for node_id in self.node_sequence):
            raise ValueError("path node IDs must be non-negative")
        if len(set(self.node_sequence)) != len(self.node_sequence):
            raise ValueError("candidate path must be simple")
        return self

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """Return consecutive directed path edges."""

        return tuple(
            zip(self.node_sequence[:-1], self.node_sequence[1:], strict=True)
        )

    @property
    def hop_count(self) -> int:
        """Return the number of directed edges."""

        return len(self.node_sequence) - 1


class PathCatalog(FrozenStrictModel):
    """Stable ordered candidate paths for one flow."""

    flow_id: int = Field(ge=0)
    paths: tuple[CandidatePath, ...] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def validate_catalog_identity(self) -> "PathCatalog":
        if any(path.flow_id != self.flow_id for path in self.paths):
            raise ValueError("every catalog path must reference the catalog flow")
        path_ids = tuple(path.path_id for path in self.paths)
        if path_ids != tuple(range(len(self.paths))):
            raise ValueError("path IDs must be contiguous and ordered from zero")
        node_sequences = tuple(path.node_sequence for path in self.paths)
        if len(set(node_sequences)) != len(node_sequences):
            raise ValueError("candidate paths must have unique node sequences")
        sort_keys = tuple((path.hop_count, path.node_sequence) for path in self.paths)
        if sort_keys != tuple(sorted(sort_keys)):
            raise ValueError("catalog must be sorted by (hop_count, node_sequence)")
        return self

    def get_path(self, path_id: int) -> CandidatePath | None:
        """Return a catalog member without repairing an invalid ID."""

        if 0 <= path_id < len(self.paths):
            candidate = self.paths[path_id]
            if candidate.path_id == path_id:
                return candidate
        return None


class SRUAllocation(FrozenStrictModel):
    """Sparse candidate allocation.

    Integer type is validated here; edge membership and positivity are evaluator
    semantics so malformed claims remain diagnosable candidates.
    """

    src: int
    dst: int
    num_srus: int

    @property
    def edge(self) -> tuple[int, int]:
        """Return the claimed directed edge."""

        return (self.src, self.dst)


class FlowDecision(FrozenStrictModel):
    """One raw, strictly typed flow decision with deferred semantics."""

    flow_id: int
    served: bool
    path_id: int
    sru_allocations: tuple[SRUAllocation, ...]


class Schedule(FrozenStrictModel):
    """A complete-candidate envelope whose semantic validation is deferred."""

    flows: tuple[FlowDecision, ...]


class EvaluatorScenario(FrozenStrictModel):
    """Complete validated state available only to the deterministic evaluator."""

    schema_version: Literal["1.0"]
    scenario_id: str = Field(min_length=1)
    node_ids: tuple[int, ...] = Field(min_length=2)
    links: tuple[DirectedLink, ...] = Field(min_length=1)
    flows: tuple[Flow, ...] = Field(min_length=1)
    path_catalogs: tuple[PathCatalog, ...] = Field(min_length=1)
    physics: PhysicsConfig
    numerics: NumericsConfig

    @model_validator(mode="after")
    def validate_scenario_relations(self) -> "EvaluatorScenario":
        if any(node_id < 0 for node_id in self.node_ids):
            raise ValueError("node IDs must be non-negative")
        if self.node_ids != tuple(sorted(set(self.node_ids))):
            raise ValueError("node IDs must be unique and deterministically sorted")

        node_set = set(self.node_ids)
        link_edges = tuple(link.edge for link in self.links)
        if len(set(link_edges)) != len(link_edges):
            raise ValueError("parallel directed links are not supported")
        if link_edges != tuple(sorted(link_edges)):
            raise ValueError("links must be deterministically sorted by (src, dst)")
        for link in self.links:
            if link.src not in node_set or link.dst not in node_set:
                raise ValueError("link endpoint is absent from node_ids")
            if link.sinr_db < self.physics.operating_sinr_db:
                raise ValueError("scenario contains a link below operating SINR")

        flow_ids = tuple(flow.flow_id for flow in self.flows)
        if flow_ids != tuple(sorted(set(flow_ids))):
            raise ValueError("flows must have unique, sorted IDs")
        for flow in self.flows:
            if flow.source not in node_set or flow.target not in node_set:
                raise ValueError("flow endpoint is absent from node_ids")

        catalog_flow_ids = tuple(catalog.flow_id for catalog in self.path_catalogs)
        if catalog_flow_ids != flow_ids:
            raise ValueError("path catalogs must occur once in the same flow-ID order")

        links = set(link_edges)
        flows = {flow.flow_id: flow for flow in self.flows}
        for catalog in self.path_catalogs:
            flow = flows[catalog.flow_id]
            for path in catalog.paths:
                if path.node_sequence[0] != flow.source:
                    raise ValueError("candidate path does not start at flow source")
                if path.node_sequence[-1] != flow.target:
                    raise ValueError("candidate path does not end at flow target")
                if any(edge not in links for edge in path.edges):
                    raise ValueError("candidate path uses an absent directed link")
        return self

    def link_by_edge(self) -> dict[tuple[int, int], DirectedLink]:
        """Build a fresh evaluator lookup without exposing mutable scenario state."""

        return {link.edge: link for link in self.links}

    def flow_by_id(self) -> dict[int, Flow]:
        """Build a fresh flow lookup."""

        return {flow.flow_id: flow for flow in self.flows}

    def catalog_by_flow_id(self) -> dict[int, PathCatalog]:
        """Build a fresh path-catalog lookup."""

        return {catalog.flow_id: catalog for catalog in self.path_catalogs}
