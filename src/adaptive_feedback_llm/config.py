"""Strict loading and validation for centralized simulation configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from yaml.constructor import ConstructorError

from adaptive_feedback_llm.exceptions import (
    ConfigurationFileError,
    ConfigurationValidationError,
)

RELIABILITY_SLOPE_PER_DB = 0.5994738181995927


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys at every mapping depth."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    """Construct one mapping without PyYAML's silent last-key-wins behavior."""

    if not isinstance(node, yaml.MappingNode):
        raise ConstructorError(
            None,
            None,
            f"expected a mapping node, but found {node.id}",
            node.start_mark,
        )

    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            is_duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if is_duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class _FrozenModel(BaseModel):
    """Strict immutable base for reproducible configuration sections."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ConfigMetadata(_FrozenModel):
    """Use restrictions attached to a configuration profile."""

    profile_name: Literal["paper"]
    intended_use: Literal["formal_benchmark"]
    formal_benchmark_ready: Literal[True]

    @model_validator(mode="after")
    def check_readiness_matches_intended_use(self) -> "ConfigMetadata":
        """Reject contradictory formal-readiness metadata."""

        expected = self.intended_use == "formal_benchmark"
        if self.formal_benchmark_ready is not expected:
            raise ValueError(
                "formal_benchmark_ready must be true exactly when intended_use "
                "is 'formal_benchmark'"
            )
        return self


class NumericsConfig(_FrozenModel):
    """Central numerical comparison policy."""

    float_tolerance: float = Field(gt=0.0, le=1.0e-3, allow_inf_nan=False)


class PhysicsConfig(_FrozenModel):
    """Physical abstraction constants in SI units unless named as dB."""

    sru_bandwidth_hz: float = Field(gt=0.0, allow_inf_nan=False)
    operating_sinr_db: float = Field(allow_inf_nan=False)
    reliability_slope_per_db: Literal[0.5994738181995927]
    implementation_efficiency: Literal[1.0]


class PathsConfig(_FrozenModel):
    """Frozen candidate-path catalogue policy."""

    candidate_count: Literal[5]


class ObjectiveConfig(_FrozenModel):
    """Frozen classification-margin coefficients."""

    delay_margin_weight: Literal[1.0]
    reliability_margin_weight: Literal[1.0]


class GuidanceConfig(_FrozenModel):
    """Frozen nonbinding static-guidance protocol."""

    hotspot_edge_load: Literal["catalog_occurrences_divided_by_capacity"]
    hotspot_path_aggregation: Literal["unweighted_sum"]
    ranking_key: Literal[
        "infeasible,total_srus,hotspot_burden,negative_class_quality,hops,nodes"
    ]


class HeuristicsConfig(_FrozenModel):
    """Shared deterministic construction rules for all three heuristics."""

    flow_order: Literal["negative_priority_then_flow_id"]
    initial_srus_per_hop: Literal[1]
    increment_srus: Literal[1]
    bottleneck_tie_break: Literal["path_edge_order"]
    rejected_path_backtracking: Literal[False]
    evaluator_repair: Literal[False]


class LLMConfig(_FrozenModel):
    """The paper Thinking(low) protocol, with no credentials or automatic retries."""

    model: Literal["deepseek-v4-flash"] = "deepseek-v4-flash"
    temperature: Literal[0.3] = 0.3
    completion_token_limit: Literal[32768] = 32768
    calls_per_round: Literal[3] = 3
    max_rounds: Literal[20] = 20
    early_stopping_patience_rounds: Literal[3] = 3
    request_timeout_s: Literal[240.0] = 240.0
    max_transport_attempts: Literal[1] = 1
    thinking_mode: Literal["enabled"] = "enabled"
    reasoning_effort: Literal["low"] = "low"
    prompt_version: Literal["1.0"] = "1.0"


class GeneticAlgorithmConfig(_FrozenModel):
    """The paper three-component GA reference and its shared kernel parameters."""

    components: dict[str, list[int]]
    independent_restarts: Literal[15]
    population_evaluations_per_scenario: Literal[5760000]
    tournament_size: Literal[3]
    elites: Literal[16]
    crossover_probability: Literal[0.9]
    random_immigrant_fraction: Literal[0.1]
    heuristic_perturbation_max_fraction: Literal[0.1]
    capacity_aware_initialization_fraction: Literal[0.3]
    flow_mutation_probability_floor: Literal[0.05]
    mutation_served_toggle_probability: Literal[0.2]
    mutation_path_change_probability: Literal[0.35]
    mutation_sru_change_probability: Literal[0.45]
    mutation_sru_subaction_rule: Literal[
        "equal_thirds_increment_decrement_uniform_resample"
    ]
    heuristic_perturbation_count_rounding: Literal["floor"]
    capacity_aware_count_rounding: Literal["round_half_up"]
    random_immigrant_count_rounding: Literal["round_half_up"]
    force_at_least_one_mutated_gene: Literal[True]
    formal_early_stopping: Literal[False]

    @model_validator(mode="after")
    def check_components(self) -> "GeneticAlgorithmConfig":
        expected = {"balanced": [256, 150], "deep-search": [512, 450], "wide-search": [768, 150]}
        if list(self.components.items()) != list(expected.items()):
            raise ValueError("GA components must match the ordered paper portfolio")
        return self


class RLConfig(_FrozenModel):
    """Paper training settings; supplied data and seeds remain explicit inputs."""

    hidden: Literal[128] = 128
    epochs: Literal[4] = 4
    batch_size: Literal[32] = 32
    learning_rate: Literal[0.0003] = 0.0003
    rollout_steps: Literal[512] = 512
    validation_interval: Literal[5000] = 5000
    environment_steps_per_seed: Literal[200000] = 200000
    seeds_per_baseline: Literal[5] = 5
    cpu_threads_per_seed: Literal[2] = 2


class StudyConfig(_FrozenModel):
    scenarios: Literal[120] = 120
    scenarios_per_difficulty: Literal[40] = 40
    llm_repeats: Literal[3] = 3
    local_work_denominator: Literal[1876511.5097109186] = 1876511.5097109186


class SimulationConfig(_FrozenModel):
    """Validated top-level project configuration."""

    schema_version: Literal["1.0"]
    metadata: ConfigMetadata
    numerics: NumericsConfig
    physics: PhysicsConfig
    paths: PathsConfig
    objective: ObjectiveConfig
    guidance: GuidanceConfig
    heuristics: HeuristicsConfig
    llm: LLMConfig
    genetic_algorithm: GeneticAlgorithmConfig
    rl: RLConfig
    study: StudyConfig


def load_config(path: str | Path) -> SimulationConfig:
    """Load a YAML file and validate it without coercion or silent correction."""

    config_path = Path(path)
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigurationFileError(
            f"Cannot read configuration file {config_path}: {exc}"
        ) from exc

    try:
        raw_data = yaml.load(raw_text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise ConfigurationFileError(
            f"Cannot decode YAML configuration {config_path}: {exc}"
        ) from exc

    if not isinstance(raw_data, dict):
        raise ConfigurationValidationError(
            f"Configuration root must be a mapping: {config_path}"
        )

    try:
        return SimulationConfig.model_validate(raw_data)
    except ValidationError as exc:
        raise ConfigurationValidationError(
            f"Configuration violates the frozen schema: {config_path}\n{exc}"
        ) from exc
