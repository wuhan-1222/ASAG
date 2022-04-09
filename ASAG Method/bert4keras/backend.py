# -*- coding: utf-8 -*-

import os, sys
from distutils.util import strtobool
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.util import nest, tf_inspect
from tensorflow.python.eager import tape
from tensorflow.python.ops.custom_gradient import _graph_mode_decorator

is_tf_keras = strtobool(os.environ.get('TF_KERAS', '0'))

if is_tf_keras:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
    sys.modules['keras'] = keras
else:
    import keras
    import keras.backend as K

do_recompute = strtobool(os.environ.get('RECOMPUTE', '0'))


def get_available_gpus():
    devices = device_lib.list_local_devices()
    devices = [x.name for x in devices if x.device_type == 'GPU']
    return devices


def gelu_erf(x):
    return 0.5 * x * (1.0 + tf.math.erf(x / np.sqrt(2.0)))


def gelu_tanh(x):
    cdf = 0.5 * (
        1.0 + K.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3))))
    )
    return x * cdf


def set_gelu(version):
    version = version.lower()
    assert version in ['erf', 'tanh'], 'gelu version must be erf or tanh'
    if version == 'erf':
        keras.utils.get_custom_objects()['gelu'] = gelu_erf
    else:
        keras.utils.get_custom_objects()['gelu'] = gelu_tanh


def infinity():
    return keras.utils.get_custom_objects().get('infinity', 1e12)


def set_infinity(value):
    keras.utils.get_custom_objects()['infinity'] = value


def piecewise_linear(t, schedule, from_zero=True):
    schedule = sorted(schedule.items())
    if from_zero and schedule[0][0] != 0:
        schedule = [(0, 0.0)] + schedule

    t = K.cast(t, K.floatx())
    x = (t * 0 + 1) * schedule[0][1]
    for i in range(len(schedule)):
        t_begin = schedule[i][0]
        x_begin = x
        if i != len(schedule) - 1:
            dx = schedule[i + 1][1] - schedule[i][1]
            dt = schedule[i + 1][0] - schedule[i][0]
            slope = 1.0 * dx / dt
            x = schedule[i][1] + slope * (t - t_begin)
        else:
            x = (t * 0 + 1) * schedule[i][1]
        x = K.switch(t >= t_begin, x, x_begin)

    return x


def search_layer(inputs, name, exclude_from=None):
    if exclude_from is None:
        exclude_from = set()

    if isinstance(inputs, keras.layers.Layer):
        layer = inputs
    else:
        layer = inputs._keras_history[0]

    if layer.name == name:
        return layer
    elif layer in exclude_from:
        return None
    else:
        exclude_from.add(layer)
        if isinstance(layer, keras.models.Model):
            model = layer
            for layer in model.layers:
                if layer.name == name:
                    return layer
        inbound_layers = layer._inbound_nodes[0].inbound_layers
        if not isinstance(inbound_layers, list):
            inbound_layers = [inbound_layers]
        if len(inbound_layers) > 0:
            for layer in inbound_layers:
                layer = search_layer(layer, name, exclude_from)
                if layer is not None:
                    return layer


def sequence_masking(x, mask, value=0, axis=None):
    if mask is None:
        return x
    else:
        x_dtype = K.dtype(x)
        if x_dtype == 'bool':
            x = K.cast(x, 'int32')
        if K.dtype(mask) != K.dtype(x):
            mask = K.cast(mask, K.dtype(x))
        if value == '-inf':
            value = -K.infinity()
        elif value == 'inf':
            value = K.infinity()
        if axis is None:
            axis = 1
        elif axis < 0:
            axis = K.ndim(x) + axis
        assert axis > 0, 'axis must be greater than 0'
        for _ in range(axis - 1):
            mask = K.expand_dims(mask, 1)
        value = K.cast(value, K.dtype(x))
        for _ in range(K.ndim(x) - K.ndim(mask)):
            mask = K.expand_dims(mask, K.ndim(mask))
        x = x * mask + value * (1 - mask)
        if x_dtype == 'bool':
            x = K.cast(x, 'bool')
        return x


def batch_gather(params, indices):
    if K.dtype(indices)[:3] != 'int':
        indices = K.cast(indices, 'int32')

    try:
        return tf.gather(params, indices, batch_dims=K.ndim(indices) - 1)
    except Exception as e1:
        try:
            return tf.batch_gather(params, indices)
        except Exception as e2:
            raise ValueError('%s\n%s\n' % (e1.message, e2.message))


def pool1d(
    x,
    pool_size,
    strides=1,
    padding='valid',
    data_format=None,
    pool_mode='max'
):
    x = K.expand_dims(x, 1)
    x = K.pool2d(
        x,
        pool_size=(1, pool_size),
        strides=(1, strides),
        padding=padding,
        data_format=data_format,
        pool_mode=pool_mode
    )
    return x[:, 0]


def divisible_temporal_padding(x, n):
    r_len = K.shape(x)[1] % n
    p_len = K.switch(r_len > 0, n - r_len, 0)
    return K.temporal_padding(x, (0, p_len))


def root_mean_square(x, axis=None, keepdims=False):
    return K.sqrt(K.mean(K.square(x), axis=axis, keepdims=keepdims))


def swish(x):
    return tf.nn.swish(x)


def leaky_relu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha=alpha)


class Sinusoidal(keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        vocab_size, depth = shape
        embeddings = np.zeros(shape)
        for pos in range(vocab_size):
            for i in range(depth // 2):
                theta = pos / np.power(10000, 2. * i / depth)
                embeddings[pos, 2 * i] = np.sin(theta)
                embeddings[pos, 2 * i + 1] = np.cos(theta)
        return embeddings


def multilabel_categorical_crossentropy(y_true, y_pred):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * K.infinity()
    y_pred_pos = y_pred - (1 - y_true) * K.infinity()
    zeros = K.zeros_like(y_pred[..., :1])
    y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
    y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
    neg_loss = tf.reduce_logsumexp(y_pred_neg, axis=-1)
    pos_loss = tf.reduce_logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss


def symbolic(f):
    return f


def graph_mode_decorator(f, *args, **kwargs):
    if tf.__version__ < '2.1':
        return _graph_mode_decorator(f, *args, **kwargs)
    else:
        return _graph_mode_decorator(f, args, kwargs)


def recompute_grad(call):
    if not do_recompute:
        return call

    def inner(self, inputs, **kwargs):
        flat_inputs = nest.flatten(inputs)
        call_args = tf_inspect.getfullargspec(call).args
        for key in ['mask', 'training']:
            if key not in call_args and key in kwargs:
                del kwargs[key]

        def kernel_call():
            return call(self, inputs, **kwargs)

        def call_and_grad(*inputs):
            if is_tf_keras:
                with tape.stop_recording():
                    outputs = kernel_call()
                    outputs = tf.identity(outputs)
            else:
                outputs = kernel_call()

            def grad_fn(doutputs, variables=None):
                watches = list(inputs)
                if variables is not None:
                    watches += list(variables)
                with tf.GradientTape() as t:
                    t.watch(watches)
                    with tf.control_dependencies([doutputs]):
                        outputs = kernel_call()
                grads = t.gradient(
                    outputs, watches, output_gradients=[doutputs]
                )
                del t
                return grads[:len(inputs)], grads[len(inputs):]

            return outputs, grad_fn

        if is_tf_keras:
            outputs, grad_fn = call_and_grad(*flat_inputs)
            flat_outputs = nest.flatten(outputs)

            def actual_grad_fn(*doutputs):
                grads = grad_fn(*doutputs, variables=self.trainable_weights)
                return grads[0] + grads[1]

            watches = flat_inputs + self.trainable_weights
            watches = [tf.convert_to_tensor(x) for x in watches]
            tape.record_operation(
                call.__name__, flat_outputs, watches, actual_grad_fn
            )
            return outputs
        else: 
            return graph_mode_decorator(call_and_grad, *flat_inputs)

    return inner

K.symbolic = getattr(K, 'symbolic', None) or symbolic

K.infinity = infinity
K.set_infinity = set_infinity
sys.modules['keras.backend'] = K

custom_objects = {
    'gelu_erf': gelu_erf,
    'gelu_tanh': gelu_tanh,
    'gelu': gelu_erf,
    'root_mean_square': root_mean_square,
    'swish': swish,
    'leaky_relu': leaky_relu,
    'Sinusoidal': Sinusoidal,
    'multilabel_categorical_crossentropy': multilabel_categorical_crossentropy,
}

keras.utils.get_custom_objects().update(custom_objects)
