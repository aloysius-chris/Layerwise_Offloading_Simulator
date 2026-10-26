"""Per-layer FLOPs and output sizes of a Keras model, in one profiling pass.

Why not the cumulative method of algorithm.py?
    algorithm.py builds, for every layer i, the sub-model input -> layer i and
    takes  FLOPs(sub-model i) - running total.  That is exact for chain
    networks (LeNet, AlexNet, VGG), but for networks with branches (Inception,
    ResNet, DenseNet, MobileNetV2, EfficientNet) the sub-model up to one branch
    does not contain the parallel branches, so the differences are wrong (even
    negative), and it needs one profiling run per layer (hundreds for
    DenseNet-121).

This module profiles the whole model ONCE with the TensorFlow profiler and
attributes every TensorFlow operation to the Keras layer whose name scope it
belongs to.  For chain networks the result is identical to algorithm.py.

FLOPs convention: the TensorFlow profiler / keras-flops convention, i.e. one
multiply-accumulate = 2 FLOPs.  Tables that quote "multiply-adds" (MACs)
report half of these numbers.
"""
import numpy as np


def _register_keras_flops_statistics():
    """Import keras_flops (it registers FLOPs for element-wise ops such as
    AddV2, Mul, ...) while tolerating statistics TensorFlow already registers."""
    from tensorflow.python.framework import ops
    original = ops.RegisterStatistics.__call__

    def tolerant(self, f):
        try:
            return original(self, f)
        except KeyError:
            return f

    ops.RegisterStatistics.__call__ = tolerant
    try:
        import keras_flops  # noqa: F401  (import has the side effect we need)
    finally:
        ops.RegisterStatistics.__call__ = original


def _output_size(layer):
    """Number of values in the layer output (batch dimension excluded)."""
    outputs = layer.output if isinstance(layer.output, (list, tuple)) else [layer.output]
    size = 0
    for tensor in outputs:
        shape = [d for d in tensor.shape[1:] if d is not None]
        size += int(np.prod(shape)) if shape else 1
    return size


def _owner_layer(op_path, layer_names):
    """Keras layer that an op belongs to.  Layer names may themselves contain
    '/' (e.g. DenseNet's 'conv1/conv'), so the longest run of consecutive path
    segments that is a layer name is taken."""
    parts = op_path.split("/")
    best = None
    for start in range(len(parts)):
        for end in range(len(parts), start, -1):
            candidate = "/".join(parts[start:end])
            if candidate in layer_names:
                if best is None or len(candidate) > len(best):
                    best = candidate
                break
    return best


def profile_layers(model):
    """Return a list of (layer_name, layer_type, flops, output_size) in the
    model's topological order (model.layers)."""
    import tensorflow as tf

    _register_keras_flops_statistics()

    # Trace the model once with batch size 1.  The traced graph has static
    # shapes, which is all the FLOP count needs, so the weights are NOT frozen
    # into constants (freezing VGG-16's 138 M parameters needs several GB).
    inputs = [tf.TensorSpec([1] + list(i.shape[1:]), i.dtype) for i in model.inputs]
    concrete = tf.function(lambda *x: model(list(x) if len(x) > 1 else x[0], training=False)).get_concrete_function(*inputs)

    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    opts["output"] = "none"
    root = tf.compat.v1.profiler.profile(graph=concrete.graph, run_meta=tf.compat.v1.RunMetadata(),
                                         cmd="scope", options=opts)

    layer_names = {layer.name for layer in model.layers}
    flops = {name: 0 for name in layer_names}
    unassigned = 0

    def visit(node):
        nonlocal unassigned
        if node.float_ops:
            owner = _owner_layer(node.name, layer_names)
            if owner is None:
                unassigned += node.float_ops
            else:
                flops[owner] += node.float_ops
        for child in node.children:
            visit(child)

    visit(root)

    rows = [(layer.name, type(layer).__name__, int(flops[layer.name]), _output_size(layer)) for layer in model.layers]
    return rows, int(root.total_float_ops), int(unassigned)


# Layers that carry no computation of their own worth scheduling separately:
# they are fused into the preceding layer (as inference engines do, e.g.
# Conv + BatchNorm + ReLU run as one kernel).  Conv, depthwise conv, dense and
# pooling layers remain separate sub-tasks; the input layer is always kept,
# because its output (the input image) is what the mobile device uploads.
LIGHT_LAYER_TYPES = {
    "BatchNormalization", "Activation", "ReLU", "Dropout", "ZeroPadding2D",
    "Reshape", "Flatten", "Rescaling", "Normalization", "Add", "Concatenate",
    "Multiply", "Lambda", "TFOpLambda", "SpatialDropout2D",
}


def fuse_light_layers(rows):
    """Merge light layers into the preceding kept layer.  The fused layer keeps
    the name of the kept layer, sums the FLOPs, and takes the output size of the
    last merged layer (the tensor that leaves the fused block).  Light layers
    that directly follow the input layer (e.g. padding, rescaling) are merged
    into the NEXT kept layer instead, so the input layer stays a pure data
    source (0 FLOPs, size = the input uploaded by the mobile device)."""
    fused = []
    carried_flops = 0          # light layers right after the input
    for name, kind, flops, size in rows:
        if fused and kind in LIGHT_LAYER_TYPES:
            if fused[-1][1] == "InputLayer":
                carried_flops += flops
            else:
                prev = fused[-1]
                fused[-1] = (prev[0], prev[1], prev[2] + flops, size)
        else:
            fused.append((name, kind, flops + carried_flops, size))
            carried_flops = 0
    return fused
