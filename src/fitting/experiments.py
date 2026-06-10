from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from fitting.model_registry import contrast_model_names

NOGSE_CONTRAST_MODELS = tuple(sorted({"free", "nogse_free_grad_offset", "rest", "tort", *contrast_model_names(family="nogse")}))
OGSE_CONTRAST_MODELS = tuple(sorted({"free", "mixed", "mixed_global", "rest", "rest_offset", "rest_offset_globC", "tort", *contrast_model_names(family="ogse")}))
NOGSE_SIGNAL_MODELS = ("free_cpmg", "free_hahn", "mixed_global", "nogse_free")
OGSE_SIGNAL_MODELS = ("monoexp", "free_ogse", "rest", "rest_offset", "ogse_free", "ogse_rest", "ogse_rest_offset")


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
        models=NOGSE_CONTRAST_MODELS,
        supports_correction=True,
    ),
    "ogse_signal_vs_g": ExperimentFamily(
        name="ogse_signal_vs_g",
        models=OGSE_SIGNAL_MODELS,
        supports_correction=True,
    ),
    "ogse_contrast_vs_g": ExperimentFamily(
        name="ogse_contrast_vs_g",
        models=OGSE_CONTRAST_MODELS,
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
