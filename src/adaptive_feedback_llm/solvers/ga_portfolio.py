"""The three-component, 45-restart GA reference used in the paper."""

from adaptive_feedback_llm.solvers.genetic_algorithm import GARunConfig, solve_genetic_algorithm

COMPONENTS = {
    "balanced": (256, 150),
    "deep-search": (512, 450),
    "wide-search": (768, 150),
}
RESTARTS = 15
POPULATION_EVALUATIONS = sum(p * g * RESTARTS for p, g in COMPONENTS.values())


COMPONENT_MASTER_SEEDS = {'balanced': 12254290203883128465526893328818864358740795929093003189184449336477797636571, 'deep-search': 110168825810603888158736038462250117621088360084977379017170314859977475896272, 'wide-search': 63415845917379566723539976326957682434806274458019277256495907228391756750367}


def component_seed(name):
    return COMPONENT_MASTER_SEEDS[name]


def solve_portfolio(scenario):
    rows = []
    for name, (population, generations) in COMPONENTS.items():
        result = solve_genetic_algorithm(
            scenario,
            master_seed=component_seed(name),
            config=GARunConfig(formal=True, population_size=population,
                               generations=generations, independent_restarts=RESTARTS),
            parallel_restarts=False,
        )
        rows.append({"component": name, **result.model_dump(mode="json")})
    # Python max retains the first exact tie, as in the frozen portfolio.
    best = max(rows, key=lambda row: row["evaluation"]["objective"])
    return {"method": "ga", "population_evaluations": POPULATION_EVALUATIONS,
            "selected_component": best["component"], "schedule": best["schedule"],
            "evaluation": best["evaluation"], "components": rows}
