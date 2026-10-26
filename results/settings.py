
task_list = [2, 4, 6, 8, 10]
task_label_list = ['100', '200', '300', '400', '500']
vm_list = [1, 2, 3, 4, 5]
vm_label_list = ['1', '2', '3', '4', '5']   # VMs per ECN (10 ECNs: 10-50 VMs in total)
vm_flops_list = [1 * 1000000000, 2 * 1000000000, 3 * 1000000000, 4 * 1000000000, 5 * 1000000000]
vm_flops_label_list = ['1G', '2G', '3G', '4G', '5G']
vm_flops = 2 * 1000000000
cloud_flops = 100 * 1000000000
vm_count = 5
ecn_count = 10
du_count = 5
md_count = 6
random_state = 1

# ---------------------------------------------------------------------------
# Deep learning models used as tasks (Table 3-2 of the thesis)
# ---------------------------------------------------------------------------
# Every task is one inference of a deep learning model, chosen at random
# (uniformly) from SELECTED_MODELS.  A task is executed layer by layer; for
# each layer the simulator needs its FLOPs and the size of its output.
#
# The layer data are read from machine_learning/model_profiles.json, produced
# once by
#     python -m machine_learning.build_model_profiles
# (TensorFlow/Keras + keras-flops, see machine_learning/model_zoo.py and
# machine_learning/layer_profiler.py).  Loading the file instead of building
# the Keras models at every run makes the experiments fast, independent of
# TensorFlow, and reproducible.
#
# Model            Input        Sub-task layers   GFLOPs (1 MAC = 2 FLOPs)
# LeNet-5          32x32x1              8            0.0008
# AlexNet          224x224x3           12            1.43
# VGG-16           224x224x3           22           30.96
# GoogleNet        224x224x3           73            3.18
# ResNet-50        224x224x3           57            7.76
# DenseNet-121     224x224x3          127            5.70
# MobileNetV2      224x224x3           55            0.62
# EfficientNet-B0  224x224x3          100            0.80
SELECTED_MODELS = [
#    "LeNet-5",
    "AlexNet",
#    "VGG-16",
    "GoogleNet",
    "ResNet-50",
    "DenseNet-121",
    "MobileNetV2",
    "EfficientNet-B0",
]

import json
import os
import sys

from machine_learning.layer import Layer


def _find_profile_file():
    candidates = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "machine_learning", "model_profiles.json")]
    candidates += [os.path.join(p, "machine_learning", "model_profiles.json") for p in sys.path if p]
    for path in candidates:
        if os.path.exists(path):
            return os.path.normpath(path)
    raise FileNotFoundError("machine_learning/model_profiles.json not found - run "
                            "'python -m machine_learning.build_model_profiles' from the project folder")


with open(_find_profile_file()) as _f:
    MODEL_PROFILES = json.load(_f)


def load_layer_list(model_name):
    """Layer objects of one model, in execution order.  The first layer is
    the model input (0 FLOPs; its size is the data uploaded by the device)."""
    layer_list = []
    for name, _layer_type, flops, output_size in MODEL_PROFILES[model_name]["layers"]:
        layer = Layer(data_size=output_size, flops_size=flops)
        layer.name = model_name + "/" + name
        layer_list.append(layer)
    return layer_list


class AlgoLayerList:
    def __init__(self):
        self.layer_list = []


algo = AlgoLayerList()

# one layer list per model; the simulator picks one at random for every task
algo.layer_list = [load_layer_list(name) for name in SELECTED_MODELS]
