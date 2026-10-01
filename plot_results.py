import csv
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_experiment(name):
    """Returns (gens, runs) where runs is a list of max-fitness arrays."""
    runs = []
    for log in sorted(glob.glob(os.path.join("results", name, "seed*", "log.csv"))):
        with open(log) as f:
            rows = list(csv.DictReader(f))
        runs.append(np.array([float(r["max_fitness"]) for r in rows]))
    if not runs:
        raise SystemExit(f"No logs found for experiment '{name}'")
    n = min(len(r) for r in runs)  # tolerate interrupted runs
    return np.arange(n), [r[:n] for r in runs]


def main(names):
    plt.figure(figsize=(9, 5.5))
    summary = []

    for name in names:
        gens, runs = load_experiment(name)
        data = np.vstack(runs)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        line, = plt.plot(gens, mean, label=f"{name} (n={len(runs)})")
        plt.fill_between(gens, mean - std, mean + std,
                         alpha=0.2, color=line.get_color())
        summary.append({
            "experiment": name,
            "seeds": len(runs),
            "final_max_fitness_mean": round(float(mean[-1]), 4),
            "final_max_fitness_std": round(float(std[-1]), 4),
            "best_fitness_any_seed": round(float(data.max()), 4),
        })

    plt.xlabel("Generation")
    plt.ylabel("Max fitness")
    plt.title("Mountain climbing: max fitness per generation")
    plt.legend()
    plt.grid(alpha=0.3)
    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", "fitness_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)

    table = os.path.join("results", "summary_table.csv")
    with open(table, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0].keys())
        w.writeheader()
        w.writerows(summary)
    print("wrote", table)


if __name__ == "__main__":
    main(sys.argv[1:] or ["baseline"])
