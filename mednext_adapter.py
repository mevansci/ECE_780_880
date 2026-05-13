import importlib
import os
import sys


def _try_import(module_path: str, symbol: str = "create_mednext_v1"):
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, symbol)
    except Exception:
        return None


def _locate_create():
    candidates = [
        "models.mednext",
        "src.models.mednext",
        "nnunet_mednext.mednext_v1",
        "MedNeXt.models.mednext",
        "standalone_inference.mednext",
        "mednext",
    ]
    for c in candidates:
        fn = _try_import(c)
        if fn is not None:
            return fn
    return None


create_mednext_v1 = _locate_create()

if create_mednext_v1 is None:
    msg = (
        "Could not find `create_mednext_v1` in repository.\n"
        "Please add the MedNeXt model creation function to one of the following locations:\n"
        " - src/models/mednext.py\n"
        " - models/mednext.py\n"
        " - nnunet_mednext.mednext_v1\n"
        " - standalone_inference/mednext.py (recommended for full standalone)\n\n"
        "As a quick fix, copy the MedNeXt model file that defines `create_mednext_v1` into\n"
        "`standalone_inference/mednext.py` and ensure it exposes `create_mednext_v1`."
    )
    raise ImportError(msg)
