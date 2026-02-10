#!/usr/bin/env python3
"""W&B Sweep agent for hyperparameter tuning.

Defines a sweep configuration, creates the sweep, and runs the agent.
The train function is a placeholder — replace with your actual training loop.

Usage:
    python sweep_agent.py
    python sweep_agent.py --count 20  # run 20 trials

Requires: wandb
"""

import argparse

import wandb


# Sweep configuration — Bayesian optimization over learning rate and batch size
SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-3,
        },
        "batch_size": {"values": [8, 16, 32, 64]},
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 1e-1,
        },
        "warmup_ratio": {"distribution": "uniform", "min": 0.0, "max": 0.2},
        "epochs": {"value": 3},
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 1,
        "eta": 3,
    },
}


def train():
    """Single training run — called by the sweep agent."""
    with wandb.init() as run:
        config = wandb.config

        # --- Replace this section with your actual training loop ---
        import math
        import random

        for epoch in range(config.epochs):
            # Simulated training metrics
            train_loss = 1.0 / (epoch + 1) + random.gauss(0, 0.1)
            val_loss = 1.0 / (epoch + 1) + 0.1 + random.gauss(0, 0.05)
            lr_factor = math.log10(config.learning_rate)

            run.log(
                {
                    "epoch": epoch,
                    "train_loss": max(0, train_loss + lr_factor * 0.1),
                    "val_loss": max(0, val_loss + lr_factor * 0.1),
                    "learning_rate": config.learning_rate,
                }
            )
        # --- End of placeholder ---


def main():
    parser = argparse.ArgumentParser(description="Run W&B sweep")
    parser.add_argument("--count", type=int, default=10, help="Number of trials")
    parser.add_argument("--project", default="sweep-demo", help="W&B project name")
    args = parser.parse_args()

    sweep_id = wandb.sweep(SWEEP_CONFIG, project=args.project)
    wandb.agent(sweep_id, function=train, count=args.count)


if __name__ == "__main__":
    main()
