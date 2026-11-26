"""The optional pre-quantized weight files published beside a Kitten ASR model."""
import os

from kitten_inference.asr._system import HUB_FILES

PRECISIONS = ("int4", "int8")
META, WEIGHTS = "quant_meta.json", "quantized.safetensors"
_FLOAT_FILES = ("model*.safetensors", "model.safetensors.index.json")


def variant(folder, precision, *, required=False):
    """Find the variant inside a model folder. A partially installed variant is an error."""
    if precision not in PRECISIONS:
        return None
    path = os.path.join(folder, precision)
    present = [os.path.isfile(os.path.join(path, name)) for name in (META, WEIGHTS)]
    if all(present):
        return path
    if required or any(present):
        raise FileNotFoundError(
            f"{folder} needs {precision}/{META} and {precision}/{WEIGHTS} for published {precision} weights."
        )
    return None


def download(snapshot_download, model, cache_dir, precision, *, required=False):
    if precision in PRECISIONS:
        common = [p for p in HUB_FILES if p not in _FLOAT_FILES]
        paths = [f"{precision}/{name}" for name in (META, WEIGHTS)]
        folder = snapshot_download(model, allow_patterns=common + paths, cache_dir=cache_dir)
        if variant(folder, precision, required=required):
            return folder
    return snapshot_download(model, allow_patterns=HUB_FILES, cache_dir=cache_dir)
