"""Deterministic unit-weight shortest simple directed path catalogs."""

from __future__ import annotations

from collections.abc import Iterable

import networkx as nx

from adaptive_feedback_llm.exceptions import InsufficientCandidatePathsError
from adaptive_feedback_llm.models import CandidatePath, DirectedLink, Flow, PathCatalog


def build_candidate_path_catalog(
    flow: Flow,
    links: Iterable[DirectedLink],
    *,
    candidate_count: int = 5,
    formal: bool = True,
) -> PathCatalog:
    """Build up to five paths sorted strictly by `(hop_count, node_sequence)`."""

    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int):
        raise TypeError("candidate_count must be an integer")
    if not 1 <= candidate_count <= 5:
        raise ValueError("candidate_count must lie in [1, 5]")
    if formal and candidate_count != 5:
        raise ValueError("formal path construction requires candidate_count=5")

    graph = nx.DiGraph()
    for link in sorted(links, key=lambda item: item.edge):
        graph.add_edge(link.src, link.dst, weight=1)

    candidates: list[tuple[int, ...]] = []
    boundary_hops: int | None = None
    try:
        generator = nx.shortest_simple_paths(
            graph,
            flow.source,
            flow.target,
            weight="weight",
        )
        for raw_path in generator:
            path = tuple(int(node) for node in raw_path)
            hop_count = len(path) - 1
            if boundary_hops is not None and hop_count > boundary_hops:
                break
            candidates.append(path)
            if len(candidates) >= candidate_count:
                boundary_hops = hop_count
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        candidates = []

    ordered = sorted(set(candidates), key=lambda nodes: (len(nodes) - 1, nodes))
    selected = ordered[:candidate_count]
    if formal and len(selected) < 5:
        raise InsufficientCandidatePathsError(
            f"flow {flow.flow_id} has {len(selected)} simple paths; formal mode requires 5"
        )
    if not selected:
        raise InsufficientCandidatePathsError(
            f"flow {flow.flow_id} has no simple directed source-target path"
        )
    return PathCatalog(
        flow_id=flow.flow_id,
        paths=tuple(
            CandidatePath(
                flow_id=flow.flow_id,
                path_id=path_id,
                node_sequence=nodes,
            )
            for path_id, nodes in enumerate(selected)
        ),
    )
