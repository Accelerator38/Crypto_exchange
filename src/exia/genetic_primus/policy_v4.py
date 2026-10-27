from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTION_TARGETS = np.asarray([-1, 0, 1], dtype=np.int8)


@dataclass(frozen=True)
class V4NetworkShape:
    inputs: int
    hidden: tuple[int, ...]
    outputs: int = 3
    maximum_active_features: int | None = None

    @property
    def dimensions(self) -> tuple[int, ...]:
        return (self.inputs, *self.hidden, self.outputs)

    @property
    def genome_size(self) -> int:
        return sum(
            left * right + right
            for left, right in zip(self.dimensions[:-1], self.dimensions[1:], strict=True)
        )


class GeneticPrimusPolicyV4:
    """Topology-explicit MLP used only by the V4 research contract."""

    def __init__(self, genome: np.ndarray, *, shape: V4NetworkShape) -> None:
        self.shape = shape
        values = np.asarray(genome, dtype=np.float32).reshape(-1)
        if len(values) != shape.genome_size:
            raise ValueError(f"invalid V4 genome size: {len(values)} != {shape.genome_size}")
        self.genome = values.copy()
        self.weights, self.biases = self.unpack(self.genome, shape)

    @staticmethod
    def unpack(
        genome: np.ndarray,
        shape: V4NetworkShape,
    ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        cursor = 0
        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        for left, right in zip(shape.dimensions[:-1], shape.dimensions[1:], strict=True):
            count = left * right
            weights.append(genome[cursor : cursor + count].reshape(left, right))
            cursor += count
            biases.append(genome[cursor : cursor + right])
            cursor += right
        if shape.maximum_active_features is not None:
            if shape.hidden or shape.outputs != 1:
                raise ValueError("hard feature masks require a binary linear overlay")
            limit = int(shape.maximum_active_features)
            if not 1 <= limit <= shape.inputs:
                raise ValueError("invalid maximum active feature count")
            importance = np.abs(weights[0][:, 0])
            keep = np.argsort(importance, kind="stable")[-limit:]
            mask = np.zeros(shape.inputs, dtype=bool)
            mask[keep] = True
            weights[0] = np.where(mask[:, None], weights[0], 0.0)
        return tuple(weights), tuple(biases)

    @staticmethod
    def pack(weights: tuple[np.ndarray, ...], biases: tuple[np.ndarray, ...]) -> np.ndarray:
        values: list[np.ndarray] = []
        for weight, bias in zip(weights, biases, strict=True):
            values.extend((weight.reshape(-1), bias.reshape(-1)))
        return np.concatenate(values).astype(np.float32)

    @classmethod
    def random(cls, rng: np.random.Generator, *, shape: V4NetworkShape) -> "GeneticPrimusPolicyV4":
        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        for left, right in zip(shape.dimensions[:-1], shape.dimensions[1:], strict=True):
            limit = np.sqrt(6.0 / (left + right))
            weights.append(rng.uniform(-limit, limit, size=(left, right)).astype(np.float32))
            biases.append(np.zeros(right, dtype=np.float32))
        return cls(cls.pack(tuple(weights), tuple(biases)), shape=shape)

    def logits(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            values = values @ weight + bias
            if index + 1 < len(self.weights):
                values = np.tanh(values)
        return values

    def target(self, features: np.ndarray) -> int:
        if self.shape.outputs != 3:
            raise ValueError("directional target requires three outputs")
        logits = self.logits(np.asarray(features, dtype=np.float32).reshape(1, -1))[0]
        return int(ACTION_TARGETS[int(np.argmax(logits))])

    def allow(self, features: np.ndarray) -> bool:
        if self.shape.outputs != 1:
            raise ValueError("binary allow requires one output")
        return self.allow_score(features) > 0.0

    def allow_score(self, features: np.ndarray) -> float:
        if self.shape.outputs != 1:
            raise ValueError("binary allow score requires one output")
        return float(self.logits(np.asarray(features, dtype=np.float32).reshape(1, -1))[0, 0])


class EnsemblePolicyV4:
    def __init__(self, genomes: tuple[np.ndarray, ...], *, shape: V4NetworkShape) -> None:
        if not genomes:
            raise ValueError("V4 ensemble requires at least one genome")
        self.members = tuple(GeneticPrimusPolicyV4(genome, shape=shape) for genome in genomes)

    def target(self, features: np.ndarray) -> int:
        if self.members[0].shape.outputs != 3:
            raise ValueError("directional target requires three outputs")
        values = np.asarray(features, dtype=np.float32).reshape(1, -1)
        mean_logits = np.mean([member.logits(values)[0] for member in self.members], axis=0)
        return int(ACTION_TARGETS[int(np.argmax(mean_logits))])

    def allow(self, features: np.ndarray) -> bool:
        if self.members[0].shape.outputs != 1:
            raise ValueError("binary allow requires one output")
        return self.allow_score(features) > 0.0

    def allow_score(self, features: np.ndarray) -> float:
        if self.members[0].shape.outputs != 1:
            raise ValueError("binary allow score requires one output")
        values = np.asarray(features, dtype=np.float32).reshape(1, -1)
        return float(np.mean([member.logits(values)[0, 0] for member in self.members]))


def ensemble_logits_v4(
    genomes: tuple[np.ndarray, ...],
    *,
    shape: V4NetworkShape,
    features: np.ndarray,
) -> np.ndarray:
    """Return mean ensemble logits on an already scaled, fixed feature anchor."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != shape.inputs:
        raise ValueError("functional drift anchor has an invalid feature shape")
    if not genomes:
        raise ValueError("functional drift requires at least one genome")
    logits = [GeneticPrimusPolicyV4(genome, shape=shape).logits(values) for genome in genomes]
    return np.mean(np.stack(logits), axis=0)


def functional_policy_drift_v4(
    challenger_genomes: tuple[np.ndarray, ...],
    incumbent_genomes: tuple[np.ndarray, ...],
    *,
    shape: V4NetworkShape,
    challenger_features: np.ndarray,
    incumbent_features: np.ndarray,
) -> dict[str, float]:
    """Compare two policies on the same raw rows after their own frozen scalers."""
    challenger = ensemble_logits_v4(
        challenger_genomes,
        shape=shape,
        features=challenger_features,
    )
    incumbent = ensemble_logits_v4(
        incumbent_genomes,
        shape=shape,
        features=incumbent_features,
    )
    if challenger.shape != incumbent.shape:
        raise ValueError("functional drift logits have different shapes")
    if shape.outputs == 1:
        challenger_bounded = np.tanh(challenger[:, 0])
        incumbent_bounded = np.tanh(incumbent[:, 0])
        logit_drift = float(np.mean(np.abs(challenger_bounded - incumbent_bounded)) / 2.0)
        action_drift = float(np.mean((challenger[:, 0] > 0.0) != (incumbent[:, 0] > 0.0)))
    else:
        def probabilities(values: np.ndarray) -> np.ndarray:
            shifted = values - values.max(axis=1, keepdims=True)
            exponent = np.exp(shifted)
            return exponent / exponent.sum(axis=1, keepdims=True)

        challenger_probability = probabilities(challenger)
        incumbent_probability = probabilities(incumbent)
        logit_drift = float(
            np.mean(0.5 * np.abs(challenger_probability - incumbent_probability).sum(axis=1))
        )
        action_drift = float(
            np.mean(np.argmax(challenger, axis=1) != np.argmax(incumbent, axis=1))
        )
    return {
        "functional_logit_drift": logit_drift,
        "functional_action_drift": action_drift,
    }


def stability_selection_v4(
    genomes: tuple[np.ndarray, ...],
    *,
    shape: V4NetworkShape,
    minimum_active_frequency: float,
    minimum_sign_agreement: float,
    active_weight_threshold: float = 0.05,
    reference_signs: np.ndarray | None = None,
) -> dict[str, object]:
    """Measure feature and coefficient-sign stability across isolated genomes."""
    if shape.hidden or shape.outputs != 1 or not genomes:
        raise ValueError("stability selection requires binary linear genomes")
    coefficients = np.stack(
        [GeneticPrimusPolicyV4(genome, shape=shape).weights[0][:, 0] for genome in genomes]
    )
    active = np.abs(coefficients) >= float(active_weight_threshold)
    active_frequency = active.mean(axis=0)
    positive = ((coefficients > 0.0) & active).sum(axis=0)
    negative = ((coefficients < 0.0) & active).sum(axis=0)
    active_count = active.sum(axis=0)
    sign_agreement = np.divide(
        np.maximum(positive, negative),
        active_count,
        out=np.zeros(shape.inputs, dtype=np.float64),
        where=active_count > 0,
    )
    consensus_sign = np.sign(positive - negative).astype(np.int8)
    stable = (active_frequency >= float(minimum_active_frequency)) & (
        sign_agreement >= float(minimum_sign_agreement)
    )
    reference_match = np.ones(shape.inputs, dtype=bool)
    if reference_signs is not None:
        expected = np.asarray(reference_signs, dtype=np.int8).reshape(-1)
        if len(expected) != shape.inputs or not np.isin(expected, (-1, 0, 1)).all():
            raise ValueError("invalid coefficient reference signs")
        constrained = expected != 0
        reference_match[constrained] = consensus_sign[constrained] == expected[constrained]
        stable &= reference_match
    return {
        "stable_feature_indices": np.flatnonzero(stable).astype(int).tolist(),
        "stable_feature_count": int(stable.sum()),
        "active_frequency": active_frequency.tolist(),
        "sign_agreement": sign_agreement.tolist(),
        "consensus_sign": consensus_sign.astype(int).tolist(),
        "reference_sign_match": reference_match.tolist(),
        "passed": bool(stable.any()),
    }


def _effective_input_mask(
    values: np.ndarray,
    *,
    shape: V4NetworkShape,
    active_weight_threshold: float,
) -> np.ndarray:
    weights, _ = GeneticPrimusPolicyV4.unpack(values, shape)
    if not shape.hidden:
        return np.any(np.abs(weights[0]) >= active_weight_threshold, axis=1)
    if len(shape.hidden) == 1:
        active_hidden = np.any(np.abs(weights[1]) >= active_weight_threshold, axis=1)
        return (
            np.any(np.abs(weights[0][:, active_hidden]) >= active_weight_threshold, axis=1)
            if active_hidden.any()
            else np.zeros(shape.inputs, dtype=bool)
        )
    raise ValueError("V4 sparsity metric permits at most one hidden layer")


def genome_sparsity_v4(
    genome: np.ndarray,
    *,
    shape: V4NetworkShape,
    active_weight_threshold: float,
) -> dict[str, float | int]:
    if active_weight_threshold <= 0.0:
        raise ValueError("active weight threshold must be positive")
    values = np.asarray(genome, dtype=np.float32).reshape(-1)
    if len(values) != shape.genome_size:
        raise ValueError("cannot measure sparsity of an invalid V4 genome")
    effective = GeneticPrimusPolicyV4.pack(*GeneticPrimusPolicyV4.unpack(values, shape))
    active_parameters = int(np.count_nonzero(np.abs(effective) >= active_weight_threshold))
    effective_inputs = _effective_input_mask(
        values, shape=shape, active_weight_threshold=active_weight_threshold
    )
    active_features = int(effective_inputs.sum())
    return {
        "parameter_count": int(shape.genome_size),
        "active_parameter_count": active_parameters,
        "parameter_sparsity": float(1.0 - active_parameters / shape.genome_size),
        "feature_count": int(shape.inputs),
        "active_feature_count": active_features,
        "feature_sparsity": float(1.0 - active_features / shape.inputs),
    }


def ensemble_sparsity_v4(
    genomes: tuple[np.ndarray, ...],
    *,
    shape: V4NetworkShape,
    active_weight_threshold: float,
) -> dict[str, float | int]:
    if not genomes:
        raise ValueError("cannot measure an empty V4 ensemble")
    values = tuple(np.asarray(genome, dtype=np.float32).reshape(-1) for genome in genomes)
    if any(len(genome) != shape.genome_size for genome in values):
        raise ValueError("cannot measure sparsity of an invalid V4 ensemble")
    active_parameters = sum(
        int(
            np.count_nonzero(
                np.abs(GeneticPrimusPolicyV4.pack(*GeneticPrimusPolicyV4.unpack(genome, shape)))
                >= active_weight_threshold
            )
        )
        for genome in values
    )
    masks = [
        _effective_input_mask(
            genome, shape=shape, active_weight_threshold=active_weight_threshold
        )
        for genome in values
    ]
    active_features = int(np.logical_or.reduce(masks).sum())
    parameter_count = shape.genome_size * len(values)
    return {
        "parameter_count": int(parameter_count),
        "active_parameter_count": int(active_parameters),
        "parameter_sparsity": float(1.0 - active_parameters / parameter_count),
        "feature_count": int(shape.inputs),
        "active_feature_count": active_features,
        "feature_sparsity": float(1.0 - active_features / shape.inputs),
        "ensemble_size": int(len(values)),
    }


def select_action_diverse_genomes_v4(
    ranked_genomes: tuple[np.ndarray, ...],
    *,
    shape: V4NetworkShape,
    anchor_features: np.ndarray,
    count: int,
    minimum_disagreement: float,
) -> tuple[tuple[np.ndarray, ...], dict[str, float | int | bool]]:
    """Greedy maximin selection over binary action signatures; no new genomes are tested."""
    if shape.outputs != 1:
        raise ValueError("action-diverse selection requires binary genomes")
    if not 1 <= count <= len(ranked_genomes):
        raise ValueError("invalid diverse ensemble size")
    anchor = np.asarray(anchor_features, dtype=np.float32)
    if anchor.ndim != 2 or anchor.shape[1] != shape.inputs or not len(anchor):
        raise ValueError("invalid diversity anchor features")
    if not 0.0 <= minimum_disagreement <= 1.0:
        raise ValueError("invalid minimum action disagreement")
    signatures = [
        np.asarray([GeneticPrimusPolicyV4(genome, shape=shape).allow(row) for row in anchor])
        for genome in ranked_genomes
    ]
    selected = [0]
    remaining = list(range(1, len(ranked_genomes)))
    selected_distances: list[float] = []
    while len(selected) < count:
        scored = [
            (
                min(float(np.mean(signatures[index] != signatures[chosen])) for chosen in selected),
                -index,
                index,
            )
            for index in remaining
        ]
        distance, _, chosen = max(scored)
        selected.append(chosen)
        selected_distances.append(distance)
        remaining.remove(chosen)
    minimum_selected = min(selected_distances, default=1.0)
    return (
        tuple(np.asarray(ranked_genomes[index], dtype=np.float32).copy() for index in selected),
        {
            "anchor_rows": int(len(anchor)),
            "selected_genomes": int(len(selected)),
            "minimum_pairwise_action_disagreement": float(minimum_selected),
            "minimum_action_disagreement_required": float(minimum_disagreement),
            "diversity_gate_passed": bool(minimum_selected >= minimum_disagreement),
        },
    )


def supervised_seed_v4(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    shape: V4NetworkShape,
    seed: int,
    epochs: int = 20,
    learning_rate: float = 0.01,
    batch_size: int = 512,
) -> GeneticPrimusPolicyV4:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.int8).reshape(-1)
    if x.ndim != 2 or x.shape[1] != shape.inputs or len(x) != len(y) or not len(x):
        raise ValueError("invalid V4 supervised seed data")
    if not set(np.unique(y)) <= {-1, 0, 1}:
        raise ValueError("V4 targets must be SHORT/FLAT/LONG")
    rng = np.random.default_rng(seed)
    policy = GeneticPrimusPolicyV4.random(rng, shape=shape)
    weights = [value.copy() for value in policy.weights]
    biases = [value.copy() for value in policy.biases]
    encoded = (y + 1).astype(np.int64)
    counts = np.bincount(encoded, minlength=3).astype(float)
    class_weight = len(encoded) / (3.0 * np.maximum(counts, 1.0))
    for _ in range(int(epochs)):
        order = rng.permutation(len(x))
        for start in range(0, len(x), int(batch_size)):
            index = order[start : start + int(batch_size)]
            xb = x[index]
            yb = encoded[index]
            activations = [xb]
            values = xb
            for layer, (weight, bias) in enumerate(zip(weights, biases, strict=True)):
                values = values @ weight + bias
                if layer + 1 < len(weights):
                    values = np.tanh(values)
                activations.append(values)
            logits = activations[-1] - activations[-1].max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            delta = probabilities
            delta[np.arange(len(yb)), yb] -= 1.0
            delta *= class_weight[yb, None]
            delta /= max(1, len(yb))
            gradients_w: list[np.ndarray] = [np.empty(0)] * len(weights)
            gradients_b: list[np.ndarray] = [np.empty(0)] * len(weights)
            for layer in range(len(weights) - 1, -1, -1):
                gradients_w[layer] = activations[layer].T @ delta
                gradients_b[layer] = delta.sum(axis=0)
                if layer:
                    hidden = activations[layer]
                    delta = (delta @ weights[layer].T) * (1.0 - hidden * hidden)
            for layer in range(len(weights)):
                weights[layer] -= float(learning_rate) * np.clip(gradients_w[layer], -5.0, 5.0)
                biases[layer] -= float(learning_rate) * np.clip(gradients_b[layer], -5.0, 5.0)
    return GeneticPrimusPolicyV4(
        GeneticPrimusPolicyV4.pack(tuple(weights), tuple(biases)),
        shape=shape,
    )


def supervised_seed_overlay_v4(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    shape: V4NetworkShape,
    seed: int,
    epochs: int = 20,
    learning_rate: float = 0.01,
    batch_size: int = 512,
) -> GeneticPrimusPolicyV4:
    """Build a deterministic binary ALLOW/ABSTAIN warm start."""
    if shape.outputs != 1 or shape.hidden:
        raise ValueError("overlay seed requires a binary linear shape")
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32).reshape(-1)
    if x.ndim != 2 or x.shape[1] != shape.inputs or len(x) != len(y) or not len(x):
        raise ValueError("invalid overlay supervised seed data")
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("overlay targets must be ABSTAIN/ALLOW")
    rng = np.random.default_rng(seed)
    weights = np.zeros(shape.inputs, dtype=np.float32)
    bias = np.float32(0.0)
    positives = max(1.0, float(y.sum()))
    negatives = max(1.0, float(len(y) - y.sum()))
    class_weights = np.where(y > 0.5, len(y) / (2.0 * positives), len(y) / (2.0 * negatives))
    for _ in range(int(epochs)):
        order = rng.permutation(len(x))
        for start in range(0, len(x), int(batch_size)):
            index = order[start : start + int(batch_size)]
            xb = x[index]
            yb = y[index]
            sample_weight = class_weights[index]
            logits = np.clip(xb @ weights + bias, -30.0, 30.0)
            error = (1.0 / (1.0 + np.exp(-logits)) - yb) * sample_weight
            weights -= float(learning_rate) * np.clip(xb.T @ error / len(index), -5.0, 5.0)
            bias -= np.float32(float(learning_rate) * np.clip(error.mean(), -5.0, 5.0))
    genome = np.concatenate([weights, np.asarray([bias], dtype=np.float32)])
    return GeneticPrimusPolicyV4(genome, shape=shape)
