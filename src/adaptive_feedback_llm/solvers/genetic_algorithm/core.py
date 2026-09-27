"""Prompt-frozen high-budget genetic algorithm with explicit small test mode."""

from __future__ import annotations

import hashlib
import math
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cmp_to_key
from typing import Literal

from pydantic import Field, model_validator

from adaptive_feedback_llm.evaluator import evaluate_schedule
from adaptive_feedback_llm.exceptions import SolverConfigurationError
from adaptive_feedback_llm.guidance import build_static_guidance
from adaptive_feedback_llm.models import (
    DiagnosticCode,
    EvaluationResult,
    EvaluatorScenario,
    FlowDecision,
    Schedule,
    SRUAllocation,
    model_to_json,
)
from adaptive_feedback_llm.models.domain import FrozenStrictModel
from adaptive_feedback_llm.solvers.common import (
    construct_path_allocation,
    reserve_allocation,
)
from adaptive_feedback_llm.solvers.heuristics import (
    solve_load_aware,
    solve_max_sinr,
    solve_priority_aware,
)


class GARunConfig(FrozenStrictModel):
    """GA run budget; only explicit non-formal runs may reduce budget fields."""

    formal: bool = True
    population_size: int = Field(default=256, ge=4)
    generations: int = Field(default=150, ge=1)
    independent_restarts: int = Field(default=15, ge=1)
    tournament_size: int = Field(default=3, ge=2)
    elites: int = Field(default=16, ge=1)
    crossover_probability: Literal[0.9] = 0.9
    random_immigrant_fraction: Literal[0.1] = 0.1
    heuristic_perturbation_max_fraction: Literal[0.1] = 0.1
    capacity_aware_initialization_fraction: Literal[0.3] = 0.3
    flow_mutation_probability_floor: Literal[0.05] = 0.05
    mutation_served_toggle_probability: Literal[0.2] = 0.2
    mutation_path_change_probability: Literal[0.35] = 0.35
    mutation_sru_change_probability: Literal[0.45] = 0.45
    mutation_sru_subaction_rule: Literal[
        "equal_thirds_increment_decrement_uniform_resample"
    ] = "equal_thirds_increment_decrement_uniform_resample"
    heuristic_perturbation_count_rounding: Literal["floor"] = "floor"
    capacity_aware_count_rounding: Literal["round_half_up"] = "round_half_up"
    random_immigrant_count_rounding: Literal["round_half_up"] = "round_half_up"
    force_at_least_one_mutated_gene: Literal[True] = True
    formal_early_stopping: Literal[False] = False

    @model_validator(mode="after")
    def validate_population_parts(self) -> "GARunConfig":
        if self.tournament_size > self.population_size:
            raise ValueError("tournament_size cannot exceed population_size")
        if self.elites >= self.population_size:
            raise ValueError("elites must be smaller than population_size")
        immigrant_count = random_immigrant_count(self.population_size)
        if self.elites + immigrant_count >= self.population_size:
            raise ValueError("elites and immigrants leave no offspring slots")
        return self


class GenerationAudit(FrozenStrictModel):
    """One generation's evaluator-backed audit metrics."""

    generation: int = Field(ge=1)
    best_feasible_objective: float = Field(ge=0.0, allow_inf_nan=False)
    feasible_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    evaluator_calls: int = Field(gt=0)
    cumulative_evaluator_calls: int = Field(gt=0)
    improved_restart_best: bool


class InitializationAudit(FrozenStrictModel):
    """Deduplicated population composition for one restart."""

    all_unserved_count: Literal[1] = 1
    unique_heuristic_seed_count: int = Field(ge=0, le=3)
    heuristic_perturbation_count: int = Field(ge=0)
    capacity_aware_count: int = Field(ge=0)
    random_candidate_count: int = Field(ge=0)
    population_size: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_total(self) -> "InitializationAudit":
        total = (
            self.all_unserved_count
            + self.unique_heuristic_seed_count
            + self.heuristic_perturbation_count
            + self.capacity_aware_count
            + self.random_candidate_count
        )
        if total != self.population_size:
            raise ValueError("initialization component counts must equal population")
        return self


class RestartAudit(FrozenStrictModel):
    """Complete audit summary for one independent restart."""

    restart_id: int = Field(ge=0)
    derived_seed: int = Field(ge=0)
    best_feasible_objective: float = Field(ge=0.0, allow_inf_nan=False)
    best_first_seen_generation: int = Field(ge=1)
    final_generation_improved: bool
    initialization: InitializationAudit
    generations: tuple[GenerationAudit, ...]


class GAResult(FrozenStrictModel):
    """Unified GA result plus restart/generation audit trail."""

    method: Literal["genetic_algorithm"] = "genetic_algorithm"
    schedule: Schedule
    evaluation: EvaluationResult
    config: GARunConfig
    master_seed: int
    parallel_restarts: bool
    heuristic_seed_methods: tuple[str, ...]
    total_evaluator_calls: int = Field(gt=0)
    restarts: tuple[RestartAudit, ...]


@dataclass(frozen=True, slots=True)
class _Individual:
    schedule: Schedule
    evaluation: EvaluationResult


_STRUCTURAL_CODES = {
    DiagnosticCode.MISSING_FLOW,
    DiagnosticCode.DUPLICATE_FLOW,
    DiagnosticCode.UNKNOWN_FLOW,
    DiagnosticCode.INVALID_UNSERVED_STATE,
    DiagnosticCode.INVALID_PATH_ID,
    DiagnosticCode.PATH_NOT_IN_CATALOG,
    DiagnosticCode.INVALID_CATALOG_PATH,
    DiagnosticCode.MISSING_ALLOCATION,
    DiagnosticCode.DUPLICATE_ALLOCATION,
    DiagnosticCode.PATH_EXTERNAL_ALLOCATION,
    DiagnosticCode.INVALID_SRU_COUNT,
}


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def heuristic_perturbation_limit(population_size: int) -> int:
    """Return the frozen at-most-10% perturbation count using floor."""

    return math.floor(population_size * 0.1)


def capacity_aware_initialization_target(population_size: int) -> int:
    """Return the approximate 30% capacity-aware target using round-half-up."""

    return _round_half_up(population_size * 0.3)


def random_immigrant_count(population_size: int) -> int:
    """Return the 10% immigrant count using round-half-up."""

    return _round_half_up(population_size * 0.1)


def derive_ga_seed(
    master_seed: int,
    scenario_id: str,
    restart_id: int,
    generation: int,
) -> int:
    payload = f"{master_seed}|{scenario_id}|{restart_id}|{generation}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _validate_formal_run(
    scenario: EvaluatorScenario,
    config: GARunConfig,
) -> None:
    if not config.formal:
        return
    expected = {
        "independent_restarts": 15,
        "tournament_size": 3,
        "elites": 16,
    }
    deviations = {
        field: getattr(config, field)
        for field, value in expected.items()
        if getattr(config, field) != value
    }
    if (config.population_size, config.generations) not in ((256, 150), (512, 450), (768, 150)):
        deviations["component_budget"] = (config.population_size, config.generations)
    if deviations:
        raise SolverConfigurationError(
            f"formal GA budget deviates from frozen configuration: {deviations}"
        )
    short = tuple(
        catalog.flow_id for catalog in scenario.path_catalogs if len(catalog.paths) != 5
    )
    if short:
        raise SolverConfigurationError(
            f"formal GA requires exactly five paths for every flow; invalid flows={short}"
        )


def _unserved(flow_id: int) -> FlowDecision:
    return FlowDecision(
        flow_id=flow_id, served=False, path_id=-1, sru_allocations=()
    )


def _random_served_decision(
    scenario: EvaluatorScenario,
    flow_id: int,
    rng: random.Random,
    *,
    exclude_path_id: int | None = None,
) -> FlowDecision:
    catalog = scenario.catalog_by_flow_id()[flow_id]
    paths = tuple(
        path for path in catalog.paths if path.path_id != exclude_path_id
    ) or catalog.paths
    path = rng.choice(paths)
    links = scenario.link_by_edge()
    return FlowDecision(
        flow_id=flow_id,
        served=True,
        path_id=path.path_id,
        sru_allocations=tuple(
            SRUAllocation(
                src=edge[0],
                dst=edge[1],
                num_srus=rng.randint(1, links[edge].capacity_srus),
            )
            for edge in path.edges
        ),
    )


def mutation_action(draw: float) -> Literal["toggle", "path", "sru"]:
    """Map a uniform draw to the frozen 20%/35%/45% action intervals."""

    if isinstance(draw, bool) or not isinstance(draw, (float, int)):
        raise TypeError("mutation draw must be a real number")
    value = float(draw)
    if not 0.0 <= value < 1.0:
        raise ValueError("mutation draw must lie in [0, 1)")
    if value < 0.2:
        return "toggle"
    if value < 0.55:
        return "path"
    return "sru"


def sru_mutation_subaction(
    draw: float,
) -> Literal["increment", "decrement", "resample"]:
    """Map a uniform draw to the three equal-probability SRU subactions."""

    if isinstance(draw, bool) or not isinstance(draw, (float, int)):
        raise TypeError("SRU mutation subaction draw must be a real number")
    value = float(draw)
    if not 0.0 <= value < 1.0:
        raise ValueError("SRU mutation subaction draw must lie in [0, 1)")
    if value < 1.0 / 3.0:
        return "increment"
    if value < 2.0 / 3.0:
        return "decrement"
    return "resample"


def _random_schedule(
    scenario: EvaluatorScenario,
    rng: random.Random,
) -> Schedule:
    return Schedule(
        flows=tuple(
            _random_served_decision(scenario, flow.flow_id, rng)
            if rng.random() < 0.5
            else _unserved(flow.flow_id)
            for flow in scenario.flows
        )
    )


def _capacity_aware_schedule(
    scenario: EvaluatorScenario,
    rng: random.Random,
) -> Schedule:
    remaining = {link.edge: link.capacity_srus for link in scenario.links}
    catalogs = scenario.catalog_by_flow_id()
    flows = list(scenario.flows)
    rng.shuffle(flows)
    decisions: dict[int, FlowDecision] = {}
    for flow in flows:
        paths = list(catalogs[flow.flow_id].paths)
        rng.shuffle(paths)
        selected = None
        for path in paths:
            construction = construct_path_allocation(
                scenario, flow, path, remaining
            )
            if construction is not None:
                selected = (path, construction)
                break
        if selected is None:
            decisions[flow.flow_id] = _unserved(flow.flow_id)
        else:
            path, construction = selected
            reserve_allocation(remaining, construction)
            decisions[flow.flow_id] = FlowDecision(
                flow_id=flow.flow_id,
                served=True,
                path_id=path.path_id,
                sru_allocations=construction.allocations,
            )
    return Schedule(flows=tuple(decisions[flow.flow_id] for flow in scenario.flows))


def _mutate_gene(
    scenario: EvaluatorScenario,
    gene: FlowDecision,
    rng: random.Random,
) -> FlowDecision:
    action = mutation_action(rng.random())
    if action == "toggle":
        if gene.served:
            return _unserved(gene.flow_id)
        return _random_served_decision(scenario, gene.flow_id, rng)
    if action == "path":
        if not gene.served:
            return gene
        catalog = scenario.catalog_by_flow_id()[gene.flow_id]
        if len(catalog.paths) < 2:
            return gene
        return _random_served_decision(
            scenario,
            gene.flow_id,
            rng,
            exclude_path_id=gene.path_id,
        )
    if not gene.served:
        return gene

    subaction = sru_mutation_subaction(rng.random())
    allocations = list(gene.sru_allocations)
    allocation_index = rng.randrange(len(allocations))
    allocation = allocations[allocation_index]
    capacity = scenario.link_by_edge()[allocation.edge].capacity_srus
    if subaction == "increment":
        if allocation.num_srus >= capacity:
            return gene
        replacement = allocation.num_srus + 1
    elif subaction == "decrement":
        if allocation.num_srus <= 1:
            return gene
        replacement = allocation.num_srus - 1
    else:
        choices = tuple(
            value for value in range(1, capacity + 1) if value != allocation.num_srus
        )
        if not choices:
            return gene
        replacement = rng.choice(choices)
    allocations[allocation_index] = SRUAllocation(
        src=allocation.src,
        dst=allocation.dst,
        num_srus=replacement,
    )
    return FlowDecision(
        flow_id=gene.flow_id,
        served=True,
        path_id=gene.path_id,
        sru_allocations=tuple(allocations),
    )


def _mutate_schedule(
    scenario: EvaluatorScenario,
    schedule: Schedule,
    rng: random.Random,
    config: GARunConfig,
) -> Schedule:
    probability = max(1.0 / len(schedule.flows), config.flow_mutation_probability_floor)
    selected = [index for index in range(len(schedule.flows)) if rng.random() < probability]
    if not selected:
        selected = [rng.randrange(len(schedule.flows))]
    genes = list(schedule.flows)
    original = tuple(genes)
    for index in selected:
        genes[index] = _mutate_gene(scenario, genes[index], rng)
    if tuple(genes) == original:
        index = selected[0]
        genes[index] = (
            _unserved(genes[index].flow_id)
            if genes[index].served
            else _random_served_decision(scenario, genes[index].flow_id, rng)
        )
    return Schedule(flows=tuple(genes))


def _crossover(
    first: Schedule,
    second: Schedule,
    rng: random.Random,
) -> Schedule:
    return Schedule(
        flows=tuple(
            left if rng.random() < 0.5 else right
            for left, right in zip(first.flows, second.flows, strict=True)
        )
    )


def ga_evaluation_order_key(result: EvaluationResult) -> tuple[float | int, ...]:
    """Return feasible-first objective or deterministic diagnostic ordering."""

    if result.feasible:
        if result.objective is None:
            raise SolverConfigurationError("feasible GA individual lacks objective")
        return (0, -result.objective)
    structural = sum(
        diagnostic.code in _STRUCTURAL_CODES for diagnostic in result.diagnostics
    )
    overloads = tuple(
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code is DiagnosticCode.LINK_CAPACITY_EXCEEDED
    )
    excess = math.fsum(
        float(diagnostic.actual) - float(diagnostic.required)
        for diagnostic in overloads
        if diagnostic.actual is not None and diagnostic.required is not None
    )
    qos_flows = {
        flow_id
        for diagnostic in result.diagnostics
        if diagnostic.code
        in {
            DiagnosticCode.DELAY_QOS_VIOLATION,
            DiagnosticCode.RELIABILITY_QOS_VIOLATION,
        }
        for flow_id in diagnostic.flow_ids
    }
    return (
        1,
        structural,
        len(overloads),
        excess,
        len(qos_flows),
        len(result.diagnostics),
    )


def _fitness_key(individual: _Individual) -> tuple[float | int, ...]:
    return ga_evaluation_order_key(individual.evaluation)


def _compare_individuals(
    left: _Individual,
    right: _Individual,
    tolerance: float,
) -> int:
    left_result = left.evaluation
    right_result = right.evaluation
    if left_result.feasible and not right_result.feasible:
        return -1
    if right_result.feasible and not left_result.feasible:
        return 1
    if left_result.feasible:
        if left_result.objective is None or right_result.objective is None:
            raise SolverConfigurationError("feasible GA individual lacks objective")
        if left_result.objective > right_result.objective + tolerance:
            return -1
        if right_result.objective > left_result.objective + tolerance:
            return 1
        return 0
    left_key = ga_evaluation_order_key(left_result)
    right_key = ga_evaluation_order_key(right_result)
    return -1 if left_key < right_key else (1 if right_key < left_key else 0)


def _stable_best(
    population: tuple[_Individual, ...],
    tolerance: float,
) -> _Individual:
    winner = population[0]
    for challenger in population[1:]:
        if _compare_individuals(challenger, winner, tolerance) < 0:
            winner = challenger
    return winner


def _tournament(
    population: tuple[_Individual, ...],
    rng: random.Random,
    size: int,
    tolerance: float,
) -> _Individual:
    sampled = rng.sample(range(len(population)), size)
    winner = population[sampled[0]]
    for index in sampled[1:]:
        challenger = population[index]
        if _compare_individuals(challenger, winner, tolerance) < 0:
            winner = challenger
    return winner


def _deduplicated_initial_population(
    scenario: EvaluatorScenario,
    config: GARunConfig,
    rng: random.Random,
    heuristic_seeds: tuple[Schedule, ...],
) -> tuple[tuple[Schedule, ...], InitializationAudit]:
    population: list[Schedule] = []
    fingerprints: set[str] = set()

    def add(candidate: Schedule) -> bool:
        fingerprint = model_to_json(candidate, indent=None)
        if fingerprint in fingerprints:
            return False
        fingerprints.add(fingerprint)
        population.append(candidate)
        return True

    add(Schedule(flows=tuple(_unserved(flow.flow_id) for flow in scenario.flows)))
    seed_count_before = len(population)
    for seed in heuristic_seeds:
        add(seed)
    unique_heuristic_seed_count = len(population) - seed_count_before

    perturb_target = min(
        heuristic_perturbation_limit(config.population_size),
        config.population_size - len(population),
    )
    added = 0
    attempts = 0
    while added < perturb_target and attempts < config.population_size * 100:
        attempts += 1
        if add(_mutate_schedule(scenario, rng.choice(heuristic_seeds), rng, config)):
            added += 1
    perturb_added = added

    capacity_target = min(
        capacity_aware_initialization_target(config.population_size),
        config.population_size - len(population),
    )
    added = 0
    attempts = 0
    while added < capacity_target and attempts < config.population_size * 100:
        attempts += 1
        if add(_capacity_aware_schedule(scenario, rng)):
            added += 1
    capacity_added = added

    attempts = 0
    while len(population) < config.population_size:
        attempts += 1
        if add(_random_schedule(scenario, rng)):
            attempts = 0
        if attempts > config.population_size * 1000:
            raise SolverConfigurationError(
                "could not deduplicate and fill the requested GA population"
            )
    random_count = (
        len(population)
        - 1
        - unique_heuristic_seed_count
        - perturb_added
        - capacity_added
    )
    return (
        tuple(population),
        InitializationAudit(
            unique_heuristic_seed_count=unique_heuristic_seed_count,
            heuristic_perturbation_count=perturb_added,
            capacity_aware_count=capacity_added,
            random_candidate_count=random_count,
            population_size=len(population),
        ),
    )


def _evaluate_population(
    scenario: EvaluatorScenario,
    schedules: tuple[Schedule, ...],
) -> tuple[_Individual, ...]:
    return tuple(
        _Individual(schedule=schedule, evaluation=evaluate_schedule(scenario, schedule))
        for schedule in schedules
    )


def _breed_next_population(
    scenario: EvaluatorScenario,
    population: tuple[_Individual, ...],
    config: GARunConfig,
    rng: random.Random,
) -> tuple[Schedule, ...]:
    tolerance = scenario.numerics.float_tolerance
    ordered = sorted(
        population,
        key=cmp_to_key(
            lambda left, right: _compare_individuals(left, right, tolerance)
        ),
    )
    elites = [individual.schedule for individual in ordered[: config.elites]]
    immigrant_count = random_immigrant_count(config.population_size)
    offspring_count = config.population_size - len(elites) - immigrant_count
    offspring: list[Schedule] = []
    for _ in range(offspring_count):
        first = _tournament(population, rng, config.tournament_size, tolerance)
        second = _tournament(population, rng, config.tournament_size, tolerance)
        child = (
            _crossover(first.schedule, second.schedule, rng)
            if rng.random() < config.crossover_probability
            else first.schedule
        )
        offspring.append(_mutate_schedule(scenario, child, rng, config))
    immigrants = [_random_schedule(scenario, rng) for _ in range(immigrant_count)]
    result = tuple(elites + offspring + immigrants)
    if len(result) != config.population_size:
        raise SolverConfigurationError("GA population accounting is inconsistent")
    return result


@dataclass(frozen=True, slots=True)
class _RestartRun:
    best: _Individual
    audit: RestartAudit
    evaluator_calls: int


def _run_restart(
    scenario: EvaluatorScenario,
    config: GARunConfig,
    master_seed: int,
    restart_id: int,
    heuristic_schedules: tuple[Schedule, ...],
) -> _RestartRun:
    initial_seed = derive_ga_seed(master_seed, scenario.scenario_id, restart_id, -1)
    schedules, initialization_audit = _deduplicated_initial_population(
        scenario,
        config,
        random.Random(initial_seed),
        heuristic_schedules,
    )
    restart_best: _Individual | None = None
    first_seen_generation = 1
    generation_audits: list[GenerationAudit] = []
    final_improved = False
    evaluator_calls = 0
    for generation_index in range(config.generations):
        generation_number = generation_index + 1
        population = _evaluate_population(scenario, schedules)
        evaluator_calls += len(population)
        feasible = tuple(item for item in population if item.evaluation.feasible)
        generation_best = _stable_best(feasible, scenario.numerics.float_tolerance)
        improved = (
            restart_best is None
            or _compare_individuals(
                generation_best, restart_best, scenario.numerics.float_tolerance
            )
            < 0
        )
        if improved:
            restart_best = generation_best
            first_seen_generation = generation_number
        final_improved = generation_index == config.generations - 1 and improved
        generation_audits.append(
            GenerationAudit(
                generation=generation_number,
                best_feasible_objective=generation_best.evaluation.objective,
                feasible_fraction=len(feasible) / len(population),
                evaluator_calls=len(population),
                cumulative_evaluator_calls=evaluator_calls,
                improved_restart_best=improved,
            )
        )
        if generation_index < config.generations - 1:
            generation_rng = random.Random(
                derive_ga_seed(
                    master_seed,
                    scenario.scenario_id,
                    restart_id,
                    generation_index,
                )
            )
            schedules = _breed_next_population(
                scenario, population, config, generation_rng
            )

    if restart_best is None or restart_best.evaluation.objective is None:
        raise SolverConfigurationError("GA restart did not retain all-unserved feasibility")
    return _RestartRun(
        best=restart_best,
        audit=RestartAudit(
            restart_id=restart_id,
            derived_seed=initial_seed,
            best_feasible_objective=restart_best.evaluation.objective,
            best_first_seen_generation=first_seen_generation,
            final_generation_improved=final_improved,
            initialization=initialization_audit,
            generations=tuple(generation_audits),
        ),
        evaluator_calls=evaluator_calls,
    )


def solve_genetic_algorithm(
    scenario: EvaluatorScenario,
    *,
    master_seed: int,
    config: GARunConfig | None = None,
    parallel_restarts: bool = False,
) -> GAResult:
    """Run all configured restarts without early stopping."""

    run_config = config or GARunConfig()
    _validate_formal_run(scenario, run_config)
    guidance = build_static_guidance(scenario)
    heuristic_results = (
        solve_priority_aware(scenario, guidance=guidance),
        solve_max_sinr(scenario, guidance=guidance),
        solve_load_aware(scenario, guidance=guidance),
    )
    heuristic_schedules = tuple(result.schedule for result in heuristic_results)
    restart_ids = tuple(range(run_config.independent_restarts))
    if parallel_restarts and len(restart_ids) > 1:
        with ThreadPoolExecutor(max_workers=len(restart_ids)) as executor:
            restart_runs = tuple(
                executor.map(
                    lambda restart_id: _run_restart(
                        scenario,
                        run_config,
                        master_seed,
                        restart_id,
                        heuristic_schedules,
                    ),
                    restart_ids,
                )
            )
    else:
        restart_runs = tuple(
            _run_restart(
                scenario,
                run_config,
                master_seed,
                restart_id,
                heuristic_schedules,
            )
            for restart_id in restart_ids
        )

    global_best = restart_runs[0].best
    for restart_run in restart_runs[1:]:
        if _compare_individuals(
            restart_run.best, global_best, scenario.numerics.float_tolerance
        ) < 0:
            global_best = restart_run.best
    total_calls = len(heuristic_results) + sum(
        restart_run.evaluator_calls for restart_run in restart_runs
    )
    return GAResult(
        schedule=global_best.schedule,
        evaluation=global_best.evaluation,
        config=run_config,
        master_seed=master_seed,
        parallel_restarts=parallel_restarts,
        heuristic_seed_methods=tuple(result.method for result in heuristic_results),
        total_evaluator_calls=total_calls,
        restarts=tuple(restart_run.audit for restart_run in restart_runs),
    )
