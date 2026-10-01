import argparse
import csv
import json
import os
import random
import time

import numpy as np

import creature
import genome
import population
from mountain_simulation import MountainSimulation

# Baseline configuration — the control condition your experiments vary from.
CONFIG = {
    "pop_size": 20,
    "gene_count": 3,
    "generations": 30,
    "iterations": 2400,          # physics steps per evaluation (10 s at 240 fps)
    "point_mutate_rate": 0.1,
    "point_mutate_amount": 0.25,
    "shrink_mutate_rate": 0.25,
    "grow_mutate_rate": 0.1,
    "fitness_epsilon": 1e-4,     # keeps roulette selection alive when all ~0
    # --- encoding-scheme experiment knobs ---
    "freeze_prefixes": [],       # gene-name prefixes to hold fixed, e.g. ["link-", "joint-"]
    "allow_grow_shrink": True,   # False = fixed link count (true fixed body plan)
    # --- sensory ablation knob ---
    "sensing_enabled": True,     # False = sensor-off run (summit-direction bias disabled)
}


def select_parent_safe(fitmap):
    """Wraps Population.select_parent: the starter version can return None
    when rounding puts r past the last bin; fall back to the last index."""
    ind = population.Population.select_parent(fitmap)
    return ind if ind is not None else len(fitmap) - 1


def run_one_seed(cfg, out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({**cfg, "seed": seed}, f, indent=2)

    pop = population.Population(pop_size=cfg["pop_size"],
                                gene_count=cfg["gene_count"])
    
    sim = MountainSimulation(sim_id=seed, sensing_enabled=cfg["sensing_enabled"])

    # Determine which genome columns must remain unchanged.
    frozen = genome.Genome.get_frozen_indices(cfg["freeze_prefixes"])
    
    log_path = os.path.join(out_dir, "log.csv")
    with open(log_path, "w", newline="") as logf:
        writer = csv.writer(logf)
        writer.writerow(["generation", "max_fitness", "mean_fitness",
                         "std_fitness", "mean_links"])

        for gen in range(cfg["generations"]):
            t0 = time.time()
            sim.eval_population(pop, cfg["iterations"])

            # Fitness comes from fitness.compute_fitness via the simulation.
            # Clamp >= 0 for roulette selection; epsilon keeps selection
            # meaningful in early generations where nobody climbs.
            fits = [max(0.0, cr.fitness) + cfg["fitness_epsilon"]
                    for cr in pop.creatures]
            links = [len(cr.get_expanded_links()) for cr in pop.creatures]

            elite_ind = int(np.argmax(fits))
            elite = pop.creatures[elite_ind]

            writer.writerow([gen, np.max(fits), np.mean(fits),
                             np.std(fits), np.mean(links)])
            logf.flush()
            genome.Genome.to_csv(elite.dna,
                                 os.path.join(out_dir, f"elite_{gen}.csv"))
            print(f"[seed {seed}] gen {gen:3d}  "
                  f"max {np.max(fits):.3f}  mean {np.mean(fits):.3f}  "
                  f"links {np.mean(links):.1f}  ({time.time()-t0:.1f}s)")

            # --- reproduce (same operators/order as the course GA) ---
            fitmap = population.Population.get_fitness_map(fits)
            new_creatures = []
            
            for _ in range(len(pop.creatures)):
                p1 = pop.creatures[select_parent_safe(fitmap)]
                p2 = pop.creatures[select_parent_safe(fitmap)]
                
                dna = genome.Genome.crossover(p1.dna, p2.dna)
                
                dna = genome.Genome.point_mutate(
                    dna, 
                    rate=cfg["point_mutate_rate"],
                    amount=cfg["point_mutate_amount"],
                    frozen_indices=frozen  
                )
                
                if cfg["allow_grow_shrink"]:
                    dna = genome.Genome.shrink_mutate(dna, rate=cfg["shrink_mutate_rate"])
                    dna = genome.Genome.grow_mutate(dna, rate=cfg["grow_mutate_rate"])
                
                cr = creature.Creature(1)
                cr.update_dna(dna)
                new_creatures.append(cr)

            # Elitism: fittest survives unchanged
            keeper = creature.Creature(1)
            keeper.update_dna(elite.dna)
            new_creatures[0] = keeper
            pop.creatures = new_creatures

    sim.disconnect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="baseline", help="experiment name")
    ap.add_argument("--seeds", type=int, default=1, help="number of repeat runs")
    ap.add_argument("--pop", type=int, help="population size")
    ap.add_argument("--genes", type=int, help="gene count")
    ap.add_argument("--gens", type=int, help="generations")
    ap.add_argument("--point-rate", type=float, help="point mutation rate")
    ap.add_argument("--grow-rate", type=float, help="grow mutation rate")
    ap.add_argument("--shrink-rate", type=float, help="shrink mutation rate")
    ap.add_argument("--freeze", choices=["none", "morphology", "control"],default="none", help="hold a group of genes fixed (encoding-scheme experiment)")
    ap.add_argument("--fixed-body", action="store_true",help="disable grow/shrink so the link count stays fixed")
    ap.add_argument("--no-sensor", action="store_true", help="sensor-off ablation: disable the summit-direction bias")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.pop: cfg["pop_size"] = args.pop
    if args.genes: cfg["gene_count"] = args.genes
    if args.gens: cfg["generations"] = args.gens
    if args.point_rate: cfg["point_mutate_rate"] = args.point_rate
    if args.grow_rate: cfg["grow_mutate_rate"] = args.grow_rate
    if args.shrink_rate: cfg["shrink_mutate_rate"] = args.shrink_rate
    if args.freeze == "morphology": cfg["freeze_prefixes"] = ["link-", "joint-"]
    if args.freeze == "control":    cfg["freeze_prefixes"] = ["control-", "motor-", "sensor-"]
    if args.fixed_body:             cfg["allow_grow_shrink"] = False
    if args.no_sensor:              cfg["sensing_enabled"] = False

    for seed in range(args.seeds):
        out_dir = os.path.join("results", args.name, f"seed{seed}")
        run_one_seed(cfg, out_dir, seed)


if __name__ == "__main__":
    main()
