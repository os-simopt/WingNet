"""Framework-independent policy gradients with parameter-based exploration.

The objective returns a scalar *reward* to maximize. This is a small utility,
not a complete reimplementation of the paper's full hyperparameter searches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    """A bounded search dimension; integer dimensions are rounded at evaluation."""

    name: str
    lower: float
    upper: float
    init: float | None = None
    is_int: bool = False
    sigma_init: float | None = None


def pgpe_optimize(
    search_space: list[ParameterSpec],
    objective_fn: Callable[[dict[str, Any]], float],
    n_iterations: int = 50,
    population_size: int = 8,
    lr_mu: float = 0.1,
    lr_sigma: float = 0.05,
    sigma_min: float = 1e-6,
    sigma_max: float = 1e3,
    seed: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Maximize ``objective_fn`` with Gaussian search-distribution updates.

    Returns the best observed parameter dictionary and per-iteration history.
    Rewards are centered by the current population mean to reduce variance.
    """
    if not search_space:
        raise ValueError("search_space must not be empty")
    if population_size < 2 or n_iterations < 1:
        raise ValueError("population_size must be >= 2 and n_iterations >= 1")
    if not (0 < sigma_min <= sigma_max):
        raise ValueError("require 0 < sigma_min <= sigma_max")
    if lr_mu <= 0 or lr_sigma <= 0:
        raise ValueError("learning rates must be positive")

    names = [spec.name for spec in search_space]
    if len(set(names)) != len(names):
        raise ValueError("parameter names must be unique")
    if any(spec.lower >= spec.upper for spec in search_space):
        raise ValueError("each parameter needs lower < upper")

    lower = np.asarray([spec.lower for spec in search_space], dtype=np.float64)
    upper = np.asarray([spec.upper for spec in search_space], dtype=np.float64)
    mu = np.asarray(
        [spec.init if spec.init is not None else (spec.lower + spec.upper) / 2
         for spec in search_space], dtype=np.float64
    )
    sigma = np.asarray(
        [spec.sigma_init if spec.sigma_init is not None
         else (spec.upper - spec.lower) / 4 for spec in search_space],
        dtype=np.float64,
    )
    if np.any(sigma <= 0):
        raise ValueError("initial standard deviations must be positive")
    mu = np.clip(mu, lower, upper)
    sigma = np.clip(sigma, sigma_min, sigma_max)
    rng = np.random.default_rng(seed)

    best_reward = -np.inf
    best_params: dict[str, Any] = {}
    history: list[dict[str, Any]] = []
    integer_mask = np.asarray([spec.is_int for spec in search_space], dtype=bool)

    for iteration in range(n_iterations):
        # Keep the unbounded samples for score-function gradients; only the
        # arguments sent to the objective are clipped to the search domain.
        sampled = mu + sigma * rng.standard_normal((population_size, len(names)))
        rewards = np.empty(population_size, dtype=np.float64)

        for sample_index, vector in enumerate(sampled):
            clipped = np.clip(vector, lower, upper)
            params = {
                name: (int(round(value)) if integer_mask[index] else float(value))
                for index, (name, value) in enumerate(zip(names, clipped))
            }
            reward = float(objective_fn(params))
            if not np.isfinite(reward):
                raise ValueError("objective_fn must return a finite reward")
            rewards[sample_index] = reward
            if reward > best_reward:
                best_reward = reward
                best_params = params.copy()

        diff = sampled - mu
        advantage = (rewards - rewards.mean())[:, None]
        grad_mu = (advantage * diff / sigma**2).mean(axis=0)
        grad_sigma = (advantage * (diff**2 - sigma**2) / sigma**3).mean(axis=0)
        mu = np.clip(mu + lr_mu * grad_mu, lower, upper)
        sigma = np.clip(sigma + lr_sigma * grad_sigma, sigma_min, sigma_max)
        history.append({
            "iteration": iteration,
            "mu": mu.copy(),
            "sigma": sigma.copy(),
            "rewards": rewards.copy(),
            "best_reward": best_reward,
        })

    return best_params, history
