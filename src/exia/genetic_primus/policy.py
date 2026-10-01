from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTION_TARGETS = np.asarray([-1, 0, 1], dtype=np.int8)


@dataclass(frozen=True)
class NetworkShape:
    inputs: int = 30
    hidden_1: int = 32
    hidden_2: int = 16
    outputs: int = 3

    @property
    def genome_size(self) -> int:
        return (
            self.inputs * self.hidden_1
            + self.hidden_1
            + self.hidden_1 * self.hidden_2
            + self.hidden_2
            + self.hidden_2 * self.outputs
            + self.outputs
        )


class GeneticPrimusPolicy:
    """Compact target-position MLP with a flat float32 genome."""

    def __init__(self, genome: np.ndarray, *, shape: NetworkShape | None = None) -> None:
        self.shape = shape or NetworkShape()
        values = np.asarray(genome, dtype=np.float32).reshape(-1)
        if len(values) != self.shape.genome_size:
            raise ValueError(
                f"invalid Genetic_Primus genome size: {len(values)} != {self.shape.genome_size}"
            )
        self.genome = values.copy()
        self._weights = self.unpack(self.genome, self.shape)

    @staticmethod
    def unpack(genome: np.ndarray, shape: NetworkShape) -> tuple[np.ndarray, ...]:
        cursor = 0

        def take(count: int) -> np.ndarray:
            nonlocal cursor
            value = genome[cursor : cursor + count]
            cursor += count
            return value

        w1 = take(shape.inputs * shape.hidden_1).reshape(shape.inputs, shape.hidden_1)
        b1 = take(shape.hidden_1)
        w2 = take(shape.hidden_1 * shape.hidden_2).reshape(shape.hidden_1, shape.hidden_2)
        b2 = take(shape.hidden_2)
        w3 = take(shape.hidden_2 * shape.outputs).reshape(shape.hidden_2, shape.outputs)
        b3 = take(shape.outputs)
        return w1, b1, w2, b2, w3, b3

    @staticmethod
    def pack(weights: tuple[np.ndarray, ...]) -> np.ndarray:
        return np.concatenate([value.reshape(-1) for value in weights]).astype(np.float32)

    @classmethod
    def random(cls, rng: np.random.Generator, *, shape: NetworkShape | None = None) -> "GeneticPrimusPolicy":
        shape = shape or NetworkShape()

        def xavier(fan_in: int, fan_out: int) -> np.ndarray:
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)

        weights = (
            xavier(shape.inputs, shape.hidden_1),
            np.zeros(shape.hidden_1, dtype=np.float32),
            xavier(shape.hidden_1, shape.hidden_2),
            np.zeros(shape.hidden_2, dtype=np.float32),
            xavier(shape.hidden_2, shape.outputs),
            np.zeros(shape.outputs, dtype=np.float32),
        )
        return cls(cls.pack(weights), shape=shape)

    def logits(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        w1, b1, w2, b2, w3, b3 = self._weights
        hidden_1 = np.tanh(values @ w1 + b1)
        hidden_2 = np.tanh(hidden_1 @ w2 + b2)
        return hidden_2 @ w3 + b3

    def target(self, features: np.ndarray) -> int:
        logits = self.logits(np.asarray(features, dtype=np.float32).reshape(1, -1))[0]
        return int(ACTION_TARGETS[int(np.argmax(logits))])


def supervised_seed(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    epochs: int = 30,
    learning_rate: float = 0.01,
    batch_size: int = 512,
) -> GeneticPrimusPolicy:
    """Create a deterministic train-only warm start without teacher agents."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.int8).reshape(-1)
    if x.ndim != 2 or x.shape[1] != NetworkShape().inputs:
        raise ValueError("supervised seed features must match the Primus network")
    if len(x) != len(y) or not len(x):
        raise ValueError("supervised seed requires aligned non-empty samples")
    if not set(np.unique(y)) <= {-1, 0, 1}:
        raise ValueError("supervised targets must be SHORT/FLAT/LONG")
    rng = np.random.default_rng(seed)
    policy = GeneticPrimusPolicy.random(rng)
    w1, b1, w2, b2, w3, b3 = [value.copy() for value in policy._weights]
    encoded = (y + 1).astype(np.int64)
    counts = np.bincount(encoded, minlength=3).astype(np.float64)
    class_weight = len(encoded) / (3.0 * np.maximum(counts, 1.0))

    for _ in range(int(epochs)):
        order = rng.permutation(len(x))
        for start in range(0, len(x), int(batch_size)):
            index = order[start : start + int(batch_size)]
            xb = x[index]
            yb = encoded[index]
            h1 = np.tanh(xb @ w1 + b1)
            h2 = np.tanh(h1 @ w2 + b2)
            logits = h2 @ w3 + b3
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            gradient = probabilities
            gradient[np.arange(len(yb)), yb] -= 1.0
            gradient *= class_weight[yb, None]
            gradient /= max(1, len(yb))
            dw3 = h2.T @ gradient
            db3 = gradient.sum(axis=0)
            dh2 = (gradient @ w3.T) * (1.0 - h2 * h2)
            dw2 = h1.T @ dh2
            db2 = dh2.sum(axis=0)
            dh1 = (dh2 @ w2.T) * (1.0 - h1 * h1)
            dw1 = xb.T @ dh1
            db1 = dh1.sum(axis=0)
            for parameter, delta in (
                (w1, dw1), (b1, db1), (w2, dw2), (b2, db2), (w3, dw3), (b3, db3)
            ):
                parameter -= float(learning_rate) * np.clip(delta, -5.0, 5.0)
    return GeneticPrimusPolicy(GeneticPrimusPolicy.pack((w1, b1, w2, b2, w3, b3)))
