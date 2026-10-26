"""Build the layer profiles of the eight task models and save them to
machine_learning/model_profiles.json.

Run once (requires TensorFlow/Keras and keras-flops):
    python -m machine_learning.build_model_profiles
or for one model (useful on low-memory machines):
    python -m machine_learning.build_model_profiles "ResNet-50"

settings.py then loads the JSON file, so running the experiments does not
need TensorFlow and every run uses exactly the same layer data.

For every model the file stores, in execution (topological) order, one entry
per sub-task layer:  [name, keras layer type, FLOPs, output size]
    FLOPs        - FLOPs of the layer (TF profiler convention, 1 MAC = 2 FLOPs)
    output size  - number of values in the layer's output tensor; this is the
                   data a layer hands on and what is transferred if the task
                   moves after this layer.  The first entry is the input
                   layer: its size is the input the mobile device uploads.
Light layers (batch normalisation, activations, padding, add, concatenate,
...) are fused into the preceding layer; the raw per-Keras-layer counts are
stored too ("keras_layers").
"""
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from machine_learning.layer_profiler import profile_layers, fuse_light_layers  # noqa: E402
from machine_learning.model_zoo import MODELS  # noqa: E402

PROFILE_FILE = os.path.join(os.path.dirname(__file__), "model_profiles.json")


def build(names):
    import keras
    profiles = {}
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE) as f:
            profiles = json.load(f)

    for name in names:
        keras.backend.clear_session()
        model = MODELS[name]()
        rows, total, unassigned = profile_layers(model)
        fused = fuse_light_layers(rows)
        profiles[name] = {
            "keras_name": model.name,
            "input_shape": list(model.input_shape[1:]),
            "parameters": int(model.count_params()),
            "total_flops": total,
            "unassigned_flops": unassigned,
            "keras_layers": len(rows),
            "layers": [list(r) for r in fused],
        }
        print(f"{name:<16} input {tuple(model.input_shape[1:])}  keras layers {len(rows):>3}  "
              f"sub-task layers {len(fused):>3}  {total / 1e9:8.3f} GFLOPs  params {model.count_params() / 1e6:6.2f} M"
              f"{'' if unassigned == 0 else f'  (unassigned {unassigned})'}")
        with open(PROFILE_FILE, "w") as f:
            json.dump(profiles, f, indent=1)


if __name__ == "__main__":
    build(sys.argv[1:] or list(MODELS))
