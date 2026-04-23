from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

CONTRAST_MODELS = ("free", "rest", "tort")
NOGSE_SIGNAL_MODELS = ("free_cpmg", "free_hahn")


@dataclass(frozen=True)
class ExperimentFamily:
    name: str
    models: tuple[str, ...]
    supports_correction: bool = True


CANONICAL_EXPERIMENTS: dict[str, ExperimentFamily] = {
    "nogse_signal_vs_g": ExperimentFamily(
        name="nogse_signal_vs_g",
        models=tuple(sorted(str(m) for m in NOGSE_SIGNAL_MODELS)),
        supports_correction=True,
    ),
    "nogse_contrast_vs_g": ExperimentFamily(
        name="nogse_contrast_vs_g",
        models=CONTRAST_MODELS,
        supports_correction=True,
    ),
    "ogse_signal_vs_g": ExperimentFamily(
        name="ogse_signal_vs_g",
        models=("monoexp",),
        supports_correction=True,
    ),
    "ogse_contrast_vs_g": ExperimentFamily(
        name="ogse_contrast_vs_g",
        models=CONTRAST_MODELS,
        supports_correction=True,
    ),
}


def canonical_experiment_names() -> tuple[str, ...]:
    return tuple(CANONICAL_EXPERIMENTS.keys())


def experiment_models(experiment: str) -> tuple[str, ...]:
    family = CANONICAL_EXPERIMENTS.get(str(experiment))
    if family is None:
        raise ValueError(
            f"Unsupported experiment family {experiment!r}. "
            f"Allowed values: {sorted(CANONICAL_EXPERIMENTS)}."
        )
    return family.models


def validate_experiment_model(experiment: str, model: str) -> str:
    models = experiment_models(experiment)
    if str(model) not in models:
        raise ValueError(
            f"Unsupported model {model!r} for experiment={experiment!r}. "
            f"Allowed values: {sorted(models)}."
        )
    return str(model)


def fit_output_name(experiment: str, model: str, *, corrected: bool) -> str:
    validate_experiment_model(experiment, model)
    suffix = "_corr" if bool(corrected) else ""
    return f"{experiment}_{model}{suffix}"


def split_all_or_values(values: Iterable[str] | None) -> list[str] | None:
    if values is None:
        return None
    out = [str(v) for v in values]
    if len(out) == 1 and out[0].upper() == "ALL":
        return None
    return out
