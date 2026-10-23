"""Stage E: FP32 / FP16 / INT8 export.

The purpose of this stage is hypothesis H5: *INT8 quantisation preserves most
of the temporal model's detection benefit while reducing latency and memory.*

The exporter produces

* an FP32 checkpoint (the reference),
* an FP16 checkpoint (weights stored in half precision),
* an INT8 checkpoint (dynamic weight quantisation of the linear and
  convolutional layers), together with a report of the on-disk size of each
  artefact and a numerical comparison of the outputs.

The calibration manifest used for the quantisation calibration set must only
contain training or validation videos; the script refuses a manifest that
mentions a test video.
"""

from __future__ import annotations

import copy
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..utils.checkpoints import load_checkpoint, save_checkpoint, sha256_file
from ..utils.experiment_registry import append_to_registry, build_record, experiment_id, results_dir
from ..utils.logging import get_logger, setup_logging

__all__ = ["quantize_model", "compare_outputs", "main"]

LOGGER = get_logger("training.quantize")

QUANTIZATION_MODES = ("fp32", "fp16", "int8")


def _assert_no_test_videos(manifest_path: str | Path | None) -> list[str]:
    """Return the video ids of a manifest and check that no test video is used."""
    if manifest_path is None:
        return []
    from ..data.video_split import read_manifest

    rows = read_manifest(manifest_path)
    test_ids = {row["video_id"] for row in rows if row.get("role") == "test"}
    used = {row["video_id"] for row in rows}
    overlap = used & test_ids
    if overlap:
        raise ValueError(
            "the quantisation calibration manifest must not contain test videos: "
            f"{sorted(overlap)[:5]}"
        )
    return sorted(used)


def quantize_model(model: torch.nn.Module, mode: str) -> torch.nn.Module:
    """Return a quantised copy of ``model``.

    Args:
        model: The model to quantise.
        mode: ``fp32``, ``fp16`` or ``int8``.
    """
    mode = mode.lower()
    if mode == "fp32":
        return model.float()
    if mode == "fp16":
        return model.half()
    if mode == "int8":
        from torch.ao.quantization import quantize_dynamic

        eager_prepare = getattr(torch.ao.quantization, "prepare_dynamic", None)
        if eager_prepare is not None:
            prepared = eager_prepare(model, inplace=False)
            return torch.ao.quantization.convert(prepared, inplace=False)
        # PyTorch >= 2.x removed the standalone prepare_dynamic/convert pair;
        # quantize_dynamic performs the same eager dynamic quantisation.
        return quantize_dynamic(model, inplace=False)
    raise ValueError(f"unknown quantisation mode: {mode!r}; expected {QUANTIZATION_MODES}")


@torch.no_grad()
def _can_run(module: torch.nn.Module, sample: torch.Tensor) -> bool:
    """Return True when ``sample`` can be pushed through ``module``."""
    try:
        _ = _logit_of(module(sample))
        return True
    except RuntimeError as exc:
        LOGGER.warning("probe forward failed: %s", exc)
        return False


def _logit_of(output: Any) -> torch.Tensor:
    """Extract the detection logit from supported output shapes."""
    if isinstance(output, dict) and "now" in output:
        return output["now"].detach()
    if isinstance(output, (tuple, list)):
        return output[0].detach()
    return output.detach()


@torch.no_grad()
def compare_outputs(
    reference: torch.nn.Module,
    candidate: torch.nn.Module,
    sample: torch.Tensor,
) -> dict[str, float]:
    """Compare the logits of two models on a sample batch.

    Temporal models may return a bare tensor, a ``(tensor, tensor)`` pair as
    ``(logits, log_variance)`` or a mapping with a ``"now"`` entry; the
    detection logit is always the quantity of interest.

    Each model receives the sample cast to its own parameter dtype, so an
    FP16 candidate and an FP32 reference can be evaluated on the same inputs
    without manual casting by the caller.
    """

    def _forward(model: torch.nn.Module) -> torch.Tensor:
        param = next(model.parameters())
        return _logit_of(model(sample.to(device=param.device, dtype=param.dtype)))

    ref = _forward(reference).float()
    out = _forward(candidate).float()
    if ref.ndim > 1:
        ref = ref.reshape(ref.shape[0], -1)
        out = out.reshape(out.shape[0], -1)
    diff = (ref - out).abs()
    denom = ref.abs().clamp(min=1e-6)
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "mean_relative_diff": float((diff / denom).mean()),
        "logit_std": float(ref.std()),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Export FP32/FP16/INT8 variants.")
    parser.add_argument("--checkpoint", required=True, help="temporal checkpoint")
    parser.add_argument("--mode", default="int8", choices=list(QUANTIZATION_MODES))
    parser.add_argument("--calibration-manifest", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--out-dir", default="results/raw")
    parser.add_argument("--registry", default="experiments/registry.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    setup_logging(level=getattr(__import__("logging"), args.log_level.upper(), 20))

    from ..models.temporal_gru import GRUConfig, TemporalGRU
    from ..models.temporal_tcn import TCNConfig, TemporalTCN

    used_videos = _assert_no_test_videos(args.calibration_manifest)

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    meta = dict(ckpt.meta)
    model_cfg = meta.get("model_config", {})
    kind = str(meta.get("model_type", "tcn")).lower()
    if kind == "gru":
        model = TemporalGRU(GRUConfig.from_dict(model_cfg))
    elif kind == "tcn":
        model = TemporalTCN(TCNConfig.from_dict(model_cfg))
    else:
        raise ValueError(f"cannot quantise temporal model type: {kind!r}")
    model.load_state_dict(ckpt.model)
    model.eval()

    sequence_length = int(meta.get("sequence_length", 16))
    feature_dim = int(model_cfg.get("input_features", model_cfg.get("feature_dimension", 16)))
    sample = torch.randn(2, sequence_length, feature_dim)

    # Reference and candidate must be independent copies: quantize_model casts
    # (and therefore mutates) the module it is given.
    reference = quantize_model(copy.deepcopy(model), "fp32")
    candidate = quantize_model(copy.deepcopy(model), args.mode)
    artifact_model = copy.deepcopy(candidate)

    # The numerical comparison runs natively (fp16) where the hardware allows,
    # and degrades to fp32 math as a last resort. compare_outputs casts each
    # model's input to its own parameter dtype, so no manual casting is needed.
    eval_reference: torch.nn.Module = reference
    eval_candidate: torch.nn.Module = candidate
    eval_sample: torch.Tensor = sample
    comparison_note: str | None = None
    if args.mode == "fp16":
        if _can_run(candidate, sample.half()):
            eval_sample = sample.half()
            comparison_note = "native fp16 forward on cpu"
        elif torch.cuda.is_available():
            eval_reference = reference.cuda()
            eval_candidate = candidate.cuda()
            eval_sample = sample.cuda().half()
            if not _can_run(eval_candidate, eval_sample):
                LOGGER.warning("half-precision CUDA forward failed; comparing in fp32")
                eval_reference = reference
                eval_candidate = candidate.float()
                eval_sample = sample
                comparison_note = (
                    "fp16 arithmetic not available on this device; comparison ran in "
                    "fp32 while the artifact stores true fp16 weights"
                )
            else:
                comparison_note = "native fp16 forward on cuda"
        else:
            comparison_note = (
                "fp16 arithmetic not available on this device; comparison ran in "
                "fp32 while the artifact stores true fp16 weights"
            )
            eval_candidate = candidate.float()

    report: dict[str, Any] = {
        "mode": args.mode,
        "source_checkpoint": str(args.checkpoint),
        "source_sha256": sha256_file(args.checkpoint),
        "calibration_manifest": str(args.calibration_manifest) if args.calibration_manifest else None,
        "n_calibration_videos": len(used_videos),
        "comparison": compare_outputs(eval_reference, eval_candidate, eval_sample),
        "comparison_note": comparison_note,
        "sizes": {},
    }

    out_dir = results_dir(args.out_dir, experiment_id(f"quantized_{args.mode}", seed=args.seed))
    target = Path(args.out) if args.out else out_dir / f"model_{args.mode}.pt"

    if args.mode == "int8":
        # TorchScript keeps the quantised operators. Newer PyTorch versions
        # deprecate eager quantisation scripting, so fall back to tracing and
        # finally to a plain state_dict export rather than aborting.
        scripted = None
        try:
            scripted = torch.jit.script(candidate)
        except Exception as exc:  # pragma: no cover - environment dependent
            LOGGER.warning("torch.jit.script failed on the INT8 model (%s); tracing", exc)
            scripted = None
        if scripted is None:
            try:
                example = torch.randn(2, sequence_length, feature_dim)
                scripted = torch.jit.trace(candidate.eval(), (example,))
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("torch.jit.trace failed (%s); saving plain state_dict", exc)
        if scripted is not None:
            scripted.save(str(target))
        else:
            save_checkpoint(
                target,
                artifact_model,
                meta={**meta, "precision": args.mode},
            )
        report["sizes"]["int8_bytes"] = target.stat().st_size
    else:
        # The FP16 checkpoint stores real half-precision weights so that the
        # on-disk size actually shrinks.
        if args.mode == "fp16" and next(artifact_model.parameters()).dtype == torch.float32:
            artifact_model = artifact_model.half()
        save_checkpoint(
            target,
            artifact_model,
            meta={**meta, "precision": args.mode},
        )
        report["sizes"][f"{args.mode}_bytes"] = target.stat().st_size

    report["sizes"]["fp32_bytes"] = Path(args.checkpoint).stat().st_size
    report["checkpoint"] = str(target)
    report["checkpoint_sha256"] = sha256_file(target)
    (out_dir / "quantization_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    record = build_record(
        exp_id=experiment_id(f"quantized_{args.mode}", seed=args.seed),
        config={"mode": args.mode, "calibration_manifest": args.calibration_manifest},
        seed=args.seed,
        device="cpu",
        precision=args.mode,
        command=" ".join(sys.argv),
        checkpoint=target,
        model=f"quantized/{kind}",
        notes="stage E",
        metrics=report["comparison"],
    )
    append_to_registry(args.registry, record)
    LOGGER.info("quantisation report: %s", json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
