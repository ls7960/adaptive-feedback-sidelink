"""Small, portable command-line interface. Every network run is explicit."""

import argparse
import json
from pathlib import Path

from adaptive_feedback_llm import __version__
from adaptive_feedback_llm.io import load_dataset, load_scenario, load_schedule, write_json


def parser():
    p = argparse.ArgumentParser(prog="sidelink")
    p.add_argument("--version", action="version", version=__version__)
    commands = p.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate a supplied schedule offline")
    evaluate.add_argument("scenario", type=Path)
    evaluate.add_argument("schedule", type=Path)
    check = commands.add_parser("check-data", help="Validate dataset identities and hashes")
    check.add_argument("manifest", type=Path)
    demo = commands.add_parser("demo", help="Run a deterministic heuristic offline")
    demo.add_argument("scenario", type=Path)
    demo.add_argument("--output", type=Path, required=True)
    llm = commands.add_parser("llm", help="Run the feedback loop or first-round baseline")
    llm.add_argument("scenario", type=Path)
    llm.add_argument("--output", type=Path, required=True)
    llm.add_argument("--mock-response", type=Path)
    llm.add_argument("--live", action="store_true", help="Allow paid external API requests")
    llm.add_argument("--one-round", action="store_true")
    ga = commands.add_parser("ga", help="Run the explicit paper GA budget on one input")
    ga.add_argument("scenario", type=Path)
    ga.add_argument("--output", type=Path, required=True)
    ga.add_argument("--full-budget", action="store_true", required=True,
                    help="Acknowledge 5,760,000 population evaluations per input")
    infer = commands.add_parser("rl-evaluate", help="Evaluate a public-format RL checkpoint")
    infer.add_argument("scenario", type=Path)
    infer.add_argument("checkpoint", type=Path)
    infer.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("rl-train", help="Train PPO on explicitly supplied disjoint datasets")
    train.add_argument("train_manifest", type=Path)
    train.add_argument("validation_manifest", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--variant", choices=["full", "sparse"], default="full")
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--steps", type=int, required=True)
    train.add_argument("--validation-interval", type=int, default=5000)
    train.add_argument("--source-only", action="store_true",
                       help="Explicit single-source full-RL design: train and select on the same one input")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == "check-data":
            metadata, scenes = load_dataset(args.manifest)
            print(json.dumps({"validated": len(scenes), "split": metadata["split"]}))
            return 0
        if args.command == "rl-train":
            from adaptive_feedback_llm.rl.training import train
            train(args.train_manifest, args.validation_manifest, args.output,
                  variant=args.variant, seed=args.seed, steps=args.steps,
                  validation_interval=args.validation_interval, source_only=args.source_only)
            print("Training completed; validation-selected checkpoint saved.")
            return 0
        scenario = load_scenario(args.scenario)
        if args.command == "evaluate":
            from adaptive_feedback_llm.evaluator import evaluate_schedule
            result = evaluate_schedule(scenario, load_schedule(args.schedule))
            print(json.dumps({"feasible": result.feasible, "objective": result.objective}))
            return 0 if result.feasible else 2
        if args.command == "llm":
            if args.live == bool(args.mock_response):
                p.error("choose exactly one of --live or --mock-response")
            from adaptive_feedback_llm.workflows import solve_llm
            result = solve_llm(scenario, args.output, live=args.live,
                               response=args.mock_response.read_text(encoding="utf-8")
                               if args.mock_response else None,
                               first_round_only=args.one_round)
            print("Result saved to the requested output directory.")
            return 3 if result["transport_halted"] else 0
        # Reserve the output before any expensive work, without overwriting prior runs.
        args.output.mkdir(parents=True, exist_ok=False)
        if args.command == "demo":
            from adaptive_feedback_llm.solvers.heuristics import solve_priority_aware
            result = solve_priority_aware(scenario).model_dump(mode="json")
        elif args.command == "ga":
            from adaptive_feedback_llm.solvers.ga_portfolio import solve_portfolio
            result = solve_portfolio(scenario)
        else:
            from adaptive_feedback_llm.rl.training import infer_checkpoint
            result = infer_checkpoint(scenario, args.checkpoint)
        write_json(args.output / "result.json", result)
        print("Result saved to the requested output directory.")
        return 0
    except Exception as error:
        # Never print a Pydantic input, HTTP body, credential, endpoint or traceback.
        print(f"Operation failed ({type(error).__name__}). Check inputs and output paths; existing outputs are not overwritten.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
