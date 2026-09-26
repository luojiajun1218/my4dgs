"""Safe, explicit S²GS frame-one snapshots for paired frame-two starts.

The checkpoint stores tensors and plain Python values only.  It deliberately
does not pickle a ``GaussianModel`` object or use the repository's incomplete
PLY writer.  Build the destination model, datasets, and loaders first, then
call :func:`load_frame1_snapshot`; that function restores RNG state last.
"""

from __future__ import annotations

from collections import OrderedDict
import os
import random
import tempfile
from typing import Any

import numpy as np
import torch
from torch import nn


SCHEMA_NAME = "s2gs_frame1_snapshot"
SCHEMA_VERSION = 1

# These are construction inputs, runtime handles, or values recreated by
# training_setup() for frame two.  Every other GaussianModel field is captured
# if it is composed of tensors and plain containers/scalars; an unknown live
# object fails closed instead of silently making an incomplete checkpoint.
_OMIT_MODEL_FIELDS = {
    "_latents",
    "latent_args",
    "model_args",
    "optimizer",
    "anchor_scheduler_args",
    "offset_scheduler_args",
    "split_generator",
    "latent_decoders",
    "scaling_activation",
    "scaling_inverse_activation",
    "covariance_activation",
    "opacity_activation",
    "inverse_opacity_activation",
    "rotation_activation",
}
_PREVIOUS_FIELDS = ("prev_atts", "prev_latents", "prev_atts_initial")
_DECODER_ATTRS = (
    "decoded_att",
    "frame_idx",
    "identity",
    "gates",
    "mask",
    "frz_div",
    "frz_layers",
)
_TAG = "__s2gs_snapshot_kind__"


def _class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _pack(value: Any) -> Any:
    """Convert state into weights-only-loadable tensors and built-ins."""
    if isinstance(value, torch.Tensor):
        return {
            _TAG: "tensor",
            "value": value.detach().to(device="cpu").clone(),
            "device": str(value.device),
            "requires_grad": bool(value.requires_grad),
        }
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object-dtype NumPy arrays are not snapshot-safe")
        return {
            _TAG: "ndarray",
            "value": value.tolist(),
            "dtype": value.dtype.str,
            "shape": list(value.shape),
        }
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, OrderedDict):
        return {
            _TAG: "ordered_dict",
            "items": [(_pack(key), _pack(item)) for key, item in value.items()],
        }
    if isinstance(value, dict):
        return {
            _TAG: "dict",
            "items": [(_pack(key), _pack(item)) for key, item in value.items()],
        }
    if isinstance(value, list):
        return {_TAG: "list", "items": [_pack(item) for item in value]}
    if isinstance(value, tuple):
        return {_TAG: "tuple", "items": [_pack(item) for item in value]}
    if isinstance(value, set):
        return {_TAG: "set", "items": [_pack(item) for item in sorted(value, key=repr)]}
    raise TypeError(
        f"Unsupported checkpoint field value of type {_class_name(value)}")


def _device_for(saved_device: str, fallback_device: torch.device) -> torch.device:
    requested = torch.device(saved_device)
    if requested.type != "cuda":
        return requested
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This snapshot contains CUDA model tensors, but CUDA is unavailable")
    # The saved ordinal can differ when a process masks visible devices.  The
    # initialized destination model is authoritative in that case.
    return fallback_device if fallback_device.type == "cuda" else torch.device("cuda:0")


def _unpack(
    value: Any,
    fallback_device: torch.device,
    reference_device: torch.device | None = None,
) -> Any:
    if not isinstance(value, dict) or _TAG not in value:
        return value
    kind = value[_TAG]
    if kind == "tensor":
        tensor = value["value"]
        if reference_device is not None:
            device = reference_device
        else:
            device = _device_for(value["device"], fallback_device)
        tensor = tensor.to(device=device)
        if value.get("requires_grad", False):
            tensor.requires_grad_(True)
        return tensor
    if kind == "ndarray":
        array = np.asarray(value["value"], dtype=value["dtype"])
        return array.reshape(value["shape"])
    if kind == "ordered_dict":
        return OrderedDict(
            (_unpack(key, fallback_device), _unpack(item, fallback_device))
            for key, item in value["items"]
        )
    if kind == "dict":
        return {
            _unpack(key, fallback_device): _unpack(item, fallback_device)
            for key, item in value["items"]
        }
    if kind == "list":
        return [_unpack(item, fallback_device) for item in value["items"]]
    if kind == "tuple":
        return tuple(_unpack(item, fallback_device) for item in value["items"])
    if kind == "set":
        return {_unpack(item, fallback_device) for item in value["items"]}
    raise ValueError(f"Unknown snapshot value kind: {kind!r}")


def _model_device(gaussians: Any) -> torch.device:
    latents = getattr(gaussians, "_latents", {})
    for value in latents.values():
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _pack_previous_state(gaussians: Any) -> dict[str, Any]:
    previous = {}
    names = tuple(getattr(gaussians, "param_names", ()))
    for field in _PREVIOUS_FIELDS:
        values = getattr(gaussians, field, None)
        if values is None:
            raise ValueError(f"GaussianModel.{field} is missing at frame-one boundary")
        if names and tuple(values.keys()) != names:
            raise ValueError(
                f"GaussianModel.{field} keys do not match param_names: "
                f"{tuple(values.keys())!r} != {names!r}")
        for name, tensor in values.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(
                    f"GaussianModel.{field}[{name!r}] is not a tensor")
        previous[field] = _pack(values)
    return previous


def _pack_decoders(gaussians: Any) -> dict[str, Any]:
    result = {}
    for name, decoder in gaussians.latent_decoders.items():
        if not isinstance(decoder, nn.Module):
            raise TypeError(
                f"Decoder {name!r} is not a torch.nn.Module; refusing an "
                "incomplete snapshot")
        attrs = {}
        for attr in _DECODER_ATTRS:
            if hasattr(decoder, attr):
                attrs[attr] = _pack(getattr(decoder, attr))
        result[name] = {
            "class": _class_name(decoder),
            "state_dict": {
                key: tensor.detach().to(device="cpu").clone()
                for key, tensor in decoder.state_dict().items()
            },
            "attributes": attrs,
            "training": bool(decoder.training),
            "requires_grad": {
                name: bool(param.requires_grad)
                for name, param in decoder.named_parameters()
            },
        }
    return result


def _pack_rng(camera_generator: random.Random, gaussians: Any) -> dict[str, Any]:
    if not hasattr(camera_generator, "getstate"):
        raise TypeError("camera_generator must provide random.Random-style getstate()")
    cuda_states = []
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_states = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]

    np_name, np_keys, np_pos, np_has_gauss, np_cached = np.random.get_state()
    split_generator = getattr(gaussians, "split_generator", None)
    split_state = None if split_generator is None else split_generator.get_state().cpu().clone()
    return {
        "python_global": _pack(random.getstate()),
        "python_camera_generator": _pack(camera_generator.getstate()),
        "numpy": {
            "algorithm": str(np_name),
            "keys": np_keys.tolist(),
            "position": int(np_pos),
            "has_gauss": int(np_has_gauss),
            "cached_gaussian": float(np_cached),
        },
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": cuda_states,
        "split_generator": split_state,
    }


def _restore_rng(
    state: dict[str, Any], camera_generator: random.Random, gaussians: Any
) -> None:
    random.setstate(_unpack(state["python_global"], torch.device("cpu")))
    camera_generator.setstate(
        _unpack(state["python_camera_generator"], torch.device("cpu")))

    np_state = state["numpy"]
    np.random.set_state((
        np_state["algorithm"],
        np.asarray(np_state["keys"], dtype=np.uint32),
        int(np_state["position"]),
        int(np_state["has_gauss"]),
        float(np_state["cached_gaussian"]),
    ))
    torch.set_rng_state(state["torch_cpu"].cpu())

    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Snapshot has CUDA RNG state, but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA device count differs from the snapshot: "
                f"saved {len(cuda_states)}, available {torch.cuda.device_count()}")
        torch.cuda.set_rng_state_all(cuda_states)

    split_state = state["split_generator"]
    split_generator = getattr(gaussians, "split_generator", None)
    if split_state is not None:
        if split_generator is None:
            raise RuntimeError(
                "Snapshot contains split-generator state but destination "
                "GaussianModel has no split_generator")
        split_generator.set_state(split_state.cpu())


def _snapshot_payload(gaussians: Any, camera_generator: random.Random) -> dict[str, Any]:
    frame_idx = int(getattr(gaussians, "frame_idx", -1))
    if frame_idx != 1:
        raise ValueError(f"Expected a frame-one model, got frame_idx={frame_idx}")
    for field in ("gate_atts", "shared_motion", "hierarchy_residual_gate"):
        if getattr(gaussians, field, None) is not None:
            raise ValueError(
                f"Cannot snapshot frame one while GaussianModel.{field} is active")

    previous = _pack_previous_state(gaussians)
    decoders = _pack_decoders(gaussians)
    latent_names = tuple(gaussians._latents.keys())
    param_names = tuple(getattr(gaussians, "param_names", latent_names))
    if latent_names != param_names:
        raise ValueError(
            f"Latent keys do not match param_names: {latent_names!r} != {param_names!r}")
    latents = {}
    for name, value in gaussians._latents.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Latent {name!r} is not a tensor")
        latents[name] = {
            "value": value.detach().cpu().clone(),
            "device": str(value.device),
            "requires_grad": bool(value.requires_grad),
        }

    model_fields = {}
    for name, value in vars(gaussians).items():
        if name in _OMIT_MODEL_FIELDS or name in _PREVIOUS_FIELDS:
            continue
        try:
            model_fields[name] = _pack(value)
        except TypeError as exc:
            raise TypeError(
                f"Cannot safely snapshot GaussianModel.{name}: {exc}") from exc

    rng = _pack_rng(camera_generator, gaussians)

    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "model_signature": {
            "param_names": list(param_names),
            "max_sh_degree": int(gaussians.max_sh_degree),
            "latent_tail_shapes": {
                name: list(value.shape[1:])
                for name, value in gaussians._latents.items()
            },
            "decoder_classes": {
                name: _class_name(decoder)
                for name, decoder in gaussians.latent_decoders.items()
            },
        },
        "latents": latents,
        "decoders": decoders,
        "previous_state": previous,
        "model_fields": model_fields,
        "rng": rng,
    }


def save_frame1_snapshot(
    path: str | os.PathLike[str],
    gaussians: Any,
    camera_generator: random.Random,
) -> dict[str, Any]:
    """Atomically save a completed frame-one Gaussian state and RNG snapshot.

    Call this at the frame boundary after train.py has copied ``prev_atts``,
    ``prev_latents``, and ``prev_atts_initial`` from frame one.  The local
    ``Random`` instance used for per-iteration camera selection must be passed
    as ``camera_generator``.
    """
    payload = _snapshot_payload(gaussians, camera_generator)
    destination = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(destination)}.", suffix=".tmp", dir=parent)
    os.close(fd)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        raise

    return {
        "path": destination,
        "schema_version": SCHEMA_VERSION,
        "frame_idx": 1,
        "num_gaussians": int(next(iter(gaussians._latents.values())).shape[0]),
    }


def _load_payload(path: str | os.PathLike[str]) -> dict[str, Any]:
    payload = torch.load(os.fspath(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("Frame-one snapshot root must be a dictionary")
    if payload.get("schema") != SCHEMA_NAME:
        raise ValueError("Not an S²GS frame-one snapshot")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported frame-one snapshot version: "
            f"{payload.get('schema_version')!r}")
    return payload


def _restore_latents(gaussians: Any, payload: dict[str, Any], device: torch.device) -> None:
    saved = payload["latents"]
    if tuple(saved.keys()) != tuple(gaussians._latents.keys()):
        raise ValueError("Destination GaussianModel latent keys differ from snapshot")
    tail_shapes = payload["model_signature"]["latent_tail_shapes"]
    for name, entry in saved.items():
        current = gaussians._latents[name]
        current_tail = tuple(current.shape[1:]) if isinstance(current, torch.Tensor) else ()
        if current_tail != tuple(tail_shapes[name]):
            raise ValueError(
                f"Latent {name!r} architecture mismatch: "
                f"destination tail shape {current_tail}, snapshot {tuple(tail_shapes[name])}")
        original_device = _device_for(entry["device"], device)
        target_device = current.device if isinstance(current, torch.Tensor) else original_device
        value = entry["value"].to(device=target_device)
        gaussians._latents[name] = nn.Parameter(
            value, requires_grad=bool(entry["requires_grad"]))


def _restore_decoders(
    gaussians: Any, payload: dict[str, Any], fallback_device: torch.device
) -> None:
    saved = payload["decoders"]
    if tuple(saved.keys()) != tuple(gaussians.latent_decoders.keys()):
        raise ValueError("Destination GaussianModel decoder keys differ from snapshot")
    for name, entry in saved.items():
        decoder = gaussians.latent_decoders[name]
        if _class_name(decoder) != entry["class"]:
            raise ValueError(
                f"Decoder {name!r} class mismatch: destination "
                f"{_class_name(decoder)}, snapshot {entry['class']}")
        decoder.load_state_dict(entry["state_dict"], strict=True)
        for attr, packed in entry["attributes"].items():
            current = getattr(decoder, attr, None)
            reference_device = current.device if isinstance(current, torch.Tensor) else None
            value = _unpack(packed, fallback_device, reference_device)
            setattr(decoder, attr, value)
        saved_requires_grad = entry["requires_grad"]
        params = dict(decoder.named_parameters())
        if tuple(saved_requires_grad.keys()) != tuple(params.keys()):
            raise ValueError(f"Decoder {name!r} parameter names differ from snapshot")
        for parameter_name, param in params.items():
            param.requires_grad_(bool(saved_requires_grad[parameter_name]))
        decoder.train(bool(entry["training"]))


def load_frame1_snapshot(
    path: str | os.PathLike[str],
    gaussians: Any,
    camera_generator: random.Random,
    *,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore a saved frame-one model into a freshly constructed model.

    Construct the matching GaussianModel and advance/recreate the train and
    test DataLoaders before calling this function.  ``restore_rng=True`` then
    restores all random streams last, so construction does not shift the
    frame-two sampling sequence.  The per-frame optimizer is cleared because
    S²GS calls ``training_setup`` at the start of frame two.
    """
    payload = _load_payload(path)
    signature = payload["model_signature"]
    if tuple(signature["param_names"]) != tuple(gaussians.param_names):
        raise ValueError("Destination GaussianModel param_names differ from snapshot")
    if int(signature["max_sh_degree"]) != int(gaussians.max_sh_degree):
        raise ValueError("Destination GaussianModel SH degree differs from snapshot")
    if tuple(payload["decoders"].keys()) != tuple(gaussians.latent_decoders.keys()):
        raise ValueError("Destination GaussianModel decoder keys differ from snapshot")
    for name, expected_class in signature["decoder_classes"].items():
        if _class_name(gaussians.latent_decoders[name]) != expected_class:
            raise ValueError(f"Destination decoder {name!r} class differs from snapshot")

    device = _model_device(gaussians)
    _restore_latents(gaussians, payload, device)
    _restore_decoders(gaussians, payload, device)

    for name, packed in payload["model_fields"].items():
        current = vars(gaussians).get(name)
        reference_device = current.device if isinstance(current, torch.Tensor) else None
        setattr(gaussians, name, _unpack(packed, device, reference_device))

    for field, packed in payload["previous_state"].items():
        current = getattr(gaussians, field, None)
        if not isinstance(current, dict):
            raise ValueError(f"Destination GaussianModel.{field} is not a mapping")
        restored = _unpack(packed, device)
        setattr(gaussians, field, restored)

    split_state = payload["rng"]["split_generator"]
    split_generator = getattr(gaussians, "split_generator", None)
    if split_state is not None:
        if split_generator is None:
            raise RuntimeError("Destination GaussianModel has no split_generator")
        split_generator.set_state(split_state.cpu())

    # Do not retain an optimizer whose parameter objects predate the latent
    # replacement.  Frame-two training_setup() makes the correct optimizer.
    gaussians.optimizer = None

    if restore_rng:
        _restore_rng(payload["rng"], camera_generator, gaussians)

    return {
        "path": os.path.abspath(os.fspath(path)),
        "schema_version": SCHEMA_VERSION,
        "frame_idx": 1,
        "num_gaussians": int(next(iter(gaussians._latents.values())).shape[0]),
        "rng_restored": bool(restore_rng),
    }
