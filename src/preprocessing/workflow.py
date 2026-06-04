"""Build and run DWI preprocessing command plans."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .commands import Command, require_tools, run_commands
from .metadata import validate_bvals_match_image, write_acqparams, write_eddy_index


BRAIN_DEFAULT_STEPS = ("denoise", "degibbs", "topup", "eddy", "eddy_qc", "bias")
PHANTOM_DEFAULT_STEPS = ("denoise", "degibbs")
SUPPORTED_STEPS = ("denoise", "degibbs", "topup", "eddy", "eddy_qc", "bias")


@dataclass(frozen=True)
class PreprocessingConfig:
    dataset: str
    subjects: tuple[str, ...]
    steps: tuple[str, ...]
    input_root: Path
    output_root: Path
    nthreads: int = 8
    session: str = "ses-T0"
    overwrite: bool = False
    dry_run: bool = False
    dwi_name: str | None = None
    bval_name: str | None = None
    bvec_name: str | None = None
    ap_json_name: str | None = None
    pa_b0_name: str | None = None
    pa_json_name: str | None = None
    ap_b0_vols: str = "0,1"
    pa_b0_vols: str = "0,1"
    slspec_name: str | None = None

    def normalized_dataset(self) -> str:
        aliases = {"brain": "brains", "brains": "brains", "phantom": "phantoms", "phantoms": "phantoms"}
        try:
            return aliases[self.dataset]
        except KeyError as exc:
            raise ValueError("--dataset must be one of: brains, brain, phantoms, phantom") from exc


def default_steps_for_dataset(dataset: str) -> tuple[str, ...]:
    normalized = {"brain": "brains", "brains": "brains", "phantom": "phantoms", "phantoms": "phantoms"}[dataset]
    return BRAIN_DEFAULT_STEPS if normalized == "brains" else PHANTOM_DEFAULT_STEPS


def supported_steps() -> tuple[str, ...]:
    return SUPPORTED_STEPS


def _subject_dwi_dir(root: Path, subject: str, session: str) -> Path:
    bids_like = root / subject / session / "dwi"
    if bids_like.exists():
        return bids_like
    return root / subject


def _subject_fmap_dir(root: Path, subject: str, session: str) -> Path:
    bids_like = root / subject / session / "fmap"
    if bids_like.exists():
        return bids_like
    return root / subject


def _name_or_default(value: str | None, subject: str, suffix: str) -> str:
    return value if value else f"{subject}_{suffix}"


def _paths_for_subject(config: PreprocessingConfig, subject: str) -> dict[str, Path]:
    dwi_dir = _subject_dwi_dir(config.input_root, subject, config.session)
    fmap_dir = _subject_fmap_dir(config.input_root, subject, config.session)
    out_dir = config.output_root / subject
    work_dir = out_dir / "preproc"

    dwi_name = _name_or_default(config.dwi_name, subject, "dwi.nii.gz")
    bval_name = _name_or_default(config.bval_name, subject, "dwi.bval")
    bvec_name = _name_or_default(config.bvec_name, subject, "dwi.bvec")
    ap_json_name = _name_or_default(config.ap_json_name, subject, "dwi.json")
    pa_b0_name = config.pa_b0_name or f"{subject}_dwi_dirPA-b0.nii.gz"
    pa_json_name = config.pa_json_name or f"{subject}_dwi_dirPA-b0.json"

    eddy_base = work_dir / "eddy" / f"{subject}_dwi_den_grc_eddy"
    topup_iout_base = work_dir / "topup" / f"{subject}_topup_unwarped_b0s"
    mask_base = work_dir / "topup" / f"{subject}_topup_unwarped_b0s_mean_brain"

    return {
        "dwi_raw": dwi_dir / dwi_name,
        "bval": dwi_dir / bval_name,
        "bvec": dwi_dir / bvec_name,
        "ap_json": dwi_dir / ap_json_name,
        "pa_b0": fmap_dir / pa_b0_name,
        "pa_json": fmap_dir / pa_json_name,
        "out_dir": out_dir,
        "work_dir": work_dir,
        "dwi_den": out_dir / f"{subject}_dwi_den.nii.gz",
        "noise": work_dir / "denoise" / f"{subject}_dwi_noise.nii.gz",
        "residuals": work_dir / "denoise" / f"{subject}_dwi_den-residuals.nii.gz",
        "dwi_den_grc": out_dir / f"{subject}_dwi_den_grc.nii.gz",
        "ap_b0": work_dir / "topup" / f"{subject}_b0_AP_raw.nii.gz",
        "pa_b0_topup": work_dir / "topup" / f"{subject}_b0_PA_raw.nii.gz",
        "b0_all": work_dir / "topup" / f"{subject}_b0s_all_topup.nii.gz",
        "acqparams": work_dir / "topup" / "acqparams.txt",
        "topup_base": work_dir / "topup" / f"{subject}_topup",
        "topup_iout": topup_iout_base,
        "topup_iout_nii": Path(f"{topup_iout_base}.nii.gz"),
        "topup_mean": work_dir / "topup" / f"{subject}_topup_unwarped_b0s_mean.nii.gz",
        "mask_base": mask_base,
        "mask": Path(f"{mask_base}_mask.nii.gz"),
        "index": work_dir / "eddy" / "index.txt",
        "eddy_out_base": eddy_base,
        "dwi_eddy": Path(f"{eddy_base}.nii.gz"),
        "rotated_bvecs": Path(f"{eddy_base}.eddy_rotated_bvecs"),
        "dwi_bias": out_dir / f"{subject}_dwi_den_grc_eddy_bias.nii.gz",
        "bias_field": work_dir / "bias" / f"{subject}_bias-field.nii.gz",
        "slspec": (dwi_dir / config.slspec_name) if config.slspec_name else Path(),
    }


def _validate_steps(config: PreprocessingConfig) -> None:
    requested = set(config.steps)
    invalid = sorted(requested - set(SUPPORTED_STEPS))
    if invalid:
        raise ValueError(f"Invalid preprocessing step(s): {', '.join(invalid)}")

    if config.normalized_dataset() == "phantoms":
        restricted = requested & {"topup", "eddy", "eddy_qc"}
        if restricted and "topup" not in requested:
            raise ValueError("Phantom eddy/eddy_qc requires topup and explicit reverse phase-encoding inputs")


def _validate_required_inputs(paths: dict[str, Path], config: PreprocessingConfig, subject: str) -> None:
    for key in ("dwi_raw", "bval", "bvec"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing required input for {subject}: {paths[key]}")

    validate_bvals_match_image(paths["dwi_raw"], paths["bval"])

    requested = set(config.steps)
    if "topup" in requested:
        for key in ("ap_json", "pa_json", "pa_b0"):
            if not paths[key].exists():
                raise FileNotFoundError(f"Missing topup input for {subject}: {paths[key]}")

    if "eddy" in requested and config.slspec_name and not paths["slspec"].exists():
        raise FileNotFoundError(f"Missing slspec file for {subject}: {paths['slspec']}")


def build_preprocessing_plan(config: PreprocessingConfig, subject: str) -> list[Command]:
    _validate_steps(config)
    paths = _paths_for_subject(config, subject)
    _validate_required_inputs(paths, config, subject)

    steps = set(config.steps)
    commands: list[Command] = []

    if "denoise" in steps:
        commands.append(Command("denoise", ("dwidenoise", "-nthreads", str(config.nthreads), "-noise", str(paths["noise"]), str(paths["dwi_raw"]), str(paths["dwi_den"])), (paths["noise"], paths["dwi_den"])))
        commands.append(Command("denoise_residuals", ("mrcalc", str(paths["dwi_raw"]), str(paths["dwi_den"]), "-subtract", str(paths["residuals"])), (paths["residuals"],)))

    if "degibbs" in steps:
        degibbs_input = paths["dwi_den"] if "denoise" in steps else paths["dwi_raw"]
        commands.append(Command("degibbs", ("mrdegibbs", str(degibbs_input), str(paths["dwi_den_grc"])), (paths["dwi_den_grc"],)))

    if "topup" in steps:
        if not config.dry_run:
            write_acqparams([paths["ap_json"], paths["ap_json"], paths["pa_json"], paths["pa_json"]], paths["acqparams"])
        commands.extend([
            Command("select_ap_b0", ("fslselectvols", "-i", str(paths["dwi_raw"]), "-o", str(paths["ap_b0"]), f"--vols={config.ap_b0_vols}"), (paths["ap_b0"],)),
            Command("select_pa_b0", ("fslselectvols", "-i", str(paths["pa_b0"]), "-o", str(paths["pa_b0_topup"]), f"--vols={config.pa_b0_vols}"), (paths["pa_b0_topup"],)),
            Command("merge_b0s", ("fslmerge", "-t", str(paths["b0_all"]), str(paths["ap_b0"]), str(paths["pa_b0_topup"])), (paths["b0_all"],)),
            Command("topup", ("topup", f"--imain={paths['b0_all']}", f"--datain={paths['acqparams']}", "--config=b02b0_1.cnf", f"--out={paths['topup_base']}", f"--fout={paths['topup_base']}-field", f"--iout={paths['topup_iout']}", f"--logout={paths['topup_base']}-log"), (paths["topup_iout_nii"],)),
            Command("topup_mean", ("fslmaths", str(paths["topup_iout_nii"]), "-Tmean", str(paths["topup_mean"])), (paths["topup_mean"],)),
            Command("topup_mask", ("bet", str(paths["topup_mean"]), str(paths["mask_base"]), "-f", "0.3", "-R", "-m"), (paths["mask"],)),
        ])

    if "eddy" in steps:
        eddy_input = paths["dwi_den_grc"] if "degibbs" in steps else paths["dwi_raw"]
        n_volumes = validate_bvals_match_image(paths["dwi_raw"], paths["bval"])
        if not config.dry_run:
            write_eddy_index(paths["index"], n_volumes)
        eddy_args = ["eddy_openmp", f"--imain={eddy_input}", f"--mask={paths['mask']}", f"--acqp={paths['acqparams']}", f"--index={paths['index']}", f"--bvecs={paths['bvec']}", f"--bvals={paths['bval']}", f"--topup={paths['topup_base']}", f"--out={paths['eddy_out_base']}", "--repol", "--ol_type=both", "--cnr_maps", "--residuals"]
        if config.slspec_name:
            eddy_args.insert(-3, f"--slspec={paths['slspec']}")
        commands.append(Command("eddy", tuple(eddy_args), (paths["dwi_eddy"], paths["rotated_bvecs"])))

    if "eddy_qc" in steps:
        commands.append(Command("eddy_qc", ("eddy_quad", str(paths["eddy_out_base"]), "-idx", str(paths["index"]), "-par", str(paths["acqparams"]), "-m", str(paths["mask"]), "-b", str(paths["bval"]))))

    if "bias" in steps:
        bias_input = paths["dwi_eddy"] if "eddy" in steps else paths["dwi_den_grc"]
        grad_bvec = paths["rotated_bvecs"] if "eddy" in steps else paths["bvec"]
        mask_args = ("-mask", str(paths["mask"])) if "topup" in steps else ()
        commands.append(Command("bias", ("dwibiascorrect", "ants", "-fslgrad", str(grad_bvec), str(paths["bval"]), *mask_args, "-bias", str(paths["bias_field"]), "-nthreads", str(config.nthreads), str(bias_input), str(paths["dwi_bias"])), (paths["dwi_bias"], paths["bias_field"])))

    return commands


def run_preprocessing(config: PreprocessingConfig) -> None:
    _validate_steps(config)
    tools = {
        "denoise": ("dwidenoise", "mrcalc"),
        "degibbs": ("mrdegibbs",),
        "topup": ("fslselectvols", "fslmerge", "topup", "fslmaths", "bet"),
        "eddy": ("eddy_openmp",),
        "eddy_qc": ("eddy_quad",),
        "bias": ("dwibiascorrect",),
    }
    required_tools = sorted({tool for step in config.steps for tool in tools[step]})
    if not config.dry_run:
        require_tools(required_tools)

    for subject in config.subjects:
        commands = build_preprocessing_plan(config, subject)
        print(f"Subject: {subject}")
        run_commands(commands, dry_run=config.dry_run, overwrite=config.overwrite)
