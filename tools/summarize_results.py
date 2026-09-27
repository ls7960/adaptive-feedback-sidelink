"""Display published paper aggregates without executing any scientific method."""

import json
from pathlib import Path


def main():
    source = Path(__file__).resolve().parents[1] / "data/published/statistics.json"
    stats = json.loads(source.read_text(encoding="utf-8"))
    for name, scopes in stats["methods"].items():
        value = scopes["overall"]["quality_percent"]
        sd = value["sample_sd"]
        spread = "fixed reference" if sd is None else f"sample SD {sd:.2f}"
        print(f"{name:20s} {value['mean']:7.2f}% ({spread})")
    print("Existing observations: 120 fixed scenarios; LLM n=3, RL n=5.")


if __name__ == "__main__":
    main()
