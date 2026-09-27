from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

import adaptive_feedback_llm.evaluator as evaluator_module
import adaptive_feedback_llm.solvers.genetic_algorithm.core as ga_core
import adaptive_feedback_llm.solvers.heuristics.core as heuristic_core
from adaptive_feedback_llm.exceptions import SolverConfigurationError
from adaptive_feedback_llm.models import EvaluatorScenario, Schedule
from adaptive_feedback_llm.solvers.genetic_algorithm import (
    GARunConfig,
    capacity_aware_initialization_target,
    ga_evaluation_order_key,
    heuristic_perturbation_limit,
    mutation_action,
    random_immigrant_count,
    solve_genetic_algorithm,
    sru_mutation_subaction,
)


def _small_config() -> GARunConfig:
    return GARunConfig(
        formal=False,
        population_size=12,
        generations=3,
        independent_restarts=3,
        tournament_size=3,
        elites=2,
    )


def test_small_nonformal_ga_is_reproducible_and_fully_audited(
    fixed_scenario: EvaluatorScenario,
) -> None:
    config = _small_config()

    first = solve_genetic_algorithm(fixed_scenario, master_seed=42, config=config)
    second = solve_genetic_algorithm(fixed_scenario, master_seed=42, config=config)

    assert first == second
    assert first.evaluation.feasible
    assert first.evaluation == evaluator_module.evaluate_schedule(
        fixed_scenario, first.schedule
    )
    assert first.total_evaluator_calls == 3 + 12 * 3 * 3
    assert first.heuristic_seed_methods == (
        "priority_aware",
        "max_sinr",
        "load_aware",
    )
    assert len(first.restarts) == 3
    assert all(len(restart.generations) == 3 for restart in first.restarts)
    assert all(
        generation.evaluator_calls == 12
        for restart in first.restarts
        for generation in restart.generations
    )
    assert all(
        restart.initialization.population_size == 12
        and restart.initialization.heuristic_perturbation_count <= 1
        and restart.initialization.capacity_aware_count <= 4
        for restart in first.restarts
    )


def test_parallel_and_serial_restarts_have_identical_ordered_results_and_audits(
    fixed_scenario: EvaluatorScenario,
) -> None:
    config = _small_config()
    serial = solve_genetic_algorithm(
        fixed_scenario, master_seed=314159, config=config, parallel_restarts=False
    )
    parallel = solve_genetic_algorithm(
        fixed_scenario, master_seed=314159, config=config, parallel_restarts=True
    )

    assert serial.schedule == parallel.schedule
    assert serial.evaluation == parallel.evaluation
    assert serial.restarts == parallel.restarts
    assert serial.total_evaluator_calls == parallel.total_evaluator_calls
    assert tuple(item.restart_id for item in parallel.restarts) == (0, 1, 2)
    assert serial.parallel_restarts is False
    assert parallel.parallel_restarts is True


def test_ga_and_heuristics_reference_the_same_evaluator_function() -> None:
    assert ga_core.evaluate_schedule is evaluator_module.evaluate_schedule
    assert heuristic_core.evaluate_schedule is evaluator_module.evaluate_schedule


def test_formal_ga_rejects_budget_deviation_before_execution(
    fixed_scenario: EvaluatorScenario,
) -> None:
    config = GARunConfig(
        formal=True,
        population_size=12,
        generations=3,
        independent_restarts=1,
        tournament_size=3,
        elites=2,
    )

    with pytest.raises(SolverConfigurationError, match="deviates"):
        solve_genetic_algorithm(fixed_scenario, master_seed=1, config=config)


def test_formal_ga_requires_five_paths_without_running_full_budget(
    fixed_scenario: EvaluatorScenario,
) -> None:
    with pytest.raises(SolverConfigurationError, match="exactly five"):
        solve_genetic_algorithm(fixed_scenario, master_seed=1, config=GARunConfig())


def test_nonformal_mode_cannot_change_frozen_mutation_protocol() -> None:
    with pytest.raises(ValidationError):
        GARunConfig(
            formal=False,
            population_size=12,
            generations=1,
            independent_restarts=1,
            tournament_size=2,
            elites=1,
            mutation_path_change_probability=0.5,
        )


def test_ga_feasibility_order_never_assigns_soft_objective(
    fixed_scenario: EvaluatorScenario,
    valid_schedule: Schedule,
) -> None:
    feasible = evaluator_module.evaluate_schedule(fixed_scenario, valid_schedule)
    invalid_schedule = Schedule(
        flows=(valid_schedule.flows[0].model_copy(update={"path_id": 99}), valid_schedule.flows[1])
    )
    infeasible = evaluator_module.evaluate_schedule(fixed_scenario, invalid_schedule)

    assert infeasible.objective is None
    assert ga_evaluation_order_key(feasible) < ga_evaluation_order_key(infeasible)


def test_mutation_action_intervals_are_exactly_20_35_45_percent() -> None:
    actions = [mutation_action((index + 0.5) / 100.0) for index in range(100)]

    assert actions.count("toggle") == 20
    assert actions.count("path") == 35
    assert actions.count("sru") == 45


def test_sru_subaction_intervals_are_equal_thirds_with_exact_boundaries() -> None:
    actions = [sru_mutation_subaction((index + 0.5) / 300.0) for index in range(300)]

    assert actions.count("increment") == 100
    assert actions.count("decrement") == 100
    assert actions.count("resample") == 100
    assert sru_mutation_subaction(0.0) == "increment"
    assert sru_mutation_subaction(math.nextafter(1 / 3, 0.0)) == "increment"
    assert sru_mutation_subaction(1 / 3) == "decrement"
    assert sru_mutation_subaction(math.nextafter(2 / 3, 0.0)) == "decrement"
    assert sru_mutation_subaction(2 / 3) == "resample"
    assert sru_mutation_subaction(math.nextafter(1.0, 0.0)) == "resample"


@pytest.mark.parametrize(
    ("current", "subaction_draw", "expected"),
    ((2, 0.0, 3), (2, 1 / 3, 1)),
)
def test_sru_increment_and_decrement_apply_only_the_sampled_subaction(
    fixed_scenario: EvaluatorScenario,
    current: int,
    subaction_draw: float,
    expected: int,
) -> None:
    class ControlledRandom:
        def __init__(self) -> None:
            self.draws = iter((0.8, subaction_draw))

        def random(self) -> float:
            return next(self.draws)

        def randrange(self, length: int) -> int:
            assert length == 1
            return 0

    gene = ga_core.FlowDecision(
        flow_id=0,
        served=True,
        path_id=0,
        sru_allocations=(ga_core.SRUAllocation(src=0, dst=1, num_srus=current),),
    )

    result = ga_core._mutate_gene(fixed_scenario, gene, ControlledRandom())

    assert result.sru_allocations[0].num_srus == expected


def test_sru_resample_is_uniform_over_legal_values_other_than_current(
    fixed_scenario: EvaluatorScenario,
) -> None:
    class ControlledRandom:
        def __init__(self) -> None:
            self.draws = iter((0.8, 0.9))

        def random(self) -> float:
            return next(self.draws)

        def randrange(self, length: int) -> int:
            assert length == 1
            return 0

        def choice(self, values):
            assert values == (1, 3)
            return values[-1]

    gene = ga_core.FlowDecision(
        flow_id=0,
        served=True,
        path_id=0,
        sru_allocations=(ga_core.SRUAllocation(src=0, dst=1, num_srus=2),),
    )

    result = ga_core._mutate_gene(fixed_scenario, gene, ControlledRandom())

    assert result.sru_allocations[0].num_srus == 3


@pytest.mark.parametrize(
    ("edge", "current", "subaction_draw"),
    (((0, 1), 3, 0.0), ((0, 1), 1, 1 / 3), ((2, 0), 1, 0.9)),
)
def test_inapplicable_sru_subaction_is_noop_without_switching(
    fixed_scenario: EvaluatorScenario,
    edge: tuple[int, int],
    current: int,
    subaction_draw: float,
) -> None:
    class ControlledRandom:
        def __init__(self) -> None:
            self.draws = iter((0.8, subaction_draw))

        def random(self) -> float:
            return next(self.draws)

        def randrange(self, length: int) -> int:
            assert length == 1
            return 0

    gene = ga_core.FlowDecision(
        flow_id=0,
        served=True,
        path_id=0,
        sru_allocations=(
            ga_core.SRUAllocation(src=edge[0], dst=edge[1], num_srus=current),
        ),
    )

    assert ga_core._mutate_gene(fixed_scenario, gene, ControlledRandom()) is gene


def test_frozen_population_count_rounding_rules() -> None:
    assert heuristic_perturbation_limit(256) == 25
    assert capacity_aware_initialization_target(256) == 77
    assert random_immigrant_count(256) == 26
    assert heuristic_perturbation_limit(12) == 1
    assert capacity_aware_initialization_target(12) == 4
    assert random_immigrant_count(12) == 1


@pytest.mark.parametrize("draw", [0.3, 0.8])
def test_inapplicable_unserved_path_and_sru_actions_are_explicit_noops(
    fixed_scenario: EvaluatorScenario,
    all_unserved_schedule: Schedule,
    draw: float,
) -> None:
    class DrawOnly:
        def random(self) -> float:
            return draw

    gene = all_unserved_schedule.flows[0]

    assert ga_core._mutate_gene(fixed_scenario, gene, DrawOnly()) is gene


def test_capacity_one_sru_action_is_noop_not_service_toggle(
    fixed_scenario: EvaluatorScenario,
) -> None:
    class DrawOnly:
        def random(self) -> float:
            return 0.8

        def randrange(self, length: int) -> int:
            assert length == 1
            return 0

    gene = ga_core.FlowDecision(
        flow_id=0,
        served=True,
        path_id=0,
        sru_allocations=(ga_core.SRUAllocation(src=2, dst=0, num_srus=1),),
    )

    assert ga_core._mutate_gene(fixed_scenario, gene, DrawOnly()) is gene


def test_force_guard_explicitly_changes_schedule_after_inapplicable_draws(
    fixed_scenario: EvaluatorScenario,
    all_unserved_schedule: Schedule,
) -> None:
    class ControlledRandom:
        def __init__(self) -> None:
            self.draws = iter((0.0, 0.9, 0.3))

        def random(self) -> float:
            return next(self.draws)

        def choice(self, values):
            return values[0]

        def randint(self, lower: int, upper: int) -> int:
            assert lower <= upper
            return lower

    mutated = ga_core._mutate_schedule(
        fixed_scenario,
        all_unserved_schedule,
        ControlledRandom(),
        _small_config(),
    )

    assert mutated != all_unserved_schedule
    assert mutated.flows[0].served is True
