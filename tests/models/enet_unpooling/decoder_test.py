from keras import backend as K
from keras.layers.advanced_activations import PReLU
import numpy as np
from src.models.enet_unpooling import decoder as sut


def generate_data(input_shape, fill_value=None, input_dtype=None):
    assert input_shape
    if not input_dtype:
        input_dtype = K.floatx()
    input_data_shape = list(input_shape)
    for i, e in enumerate(input_data_shape):
        if e is None:
            input_data_shape[i] = np.random.randint(1, 4)
    input_data_single = np.full(shape=input_data_shape, fill_value=fill_value) if fill_value else np.random.random(input_data_shape)
    input_data = (10 * input_data_single)
    input_data = input_data.astype(input_dtype)
    return input_data


def generate_tensor(input_shape, input_dtype=None):
    data = generate_data(input_shape=input_shape, input_dtype=input_dtype)
    # return K.placeholder(shape=input_shape)
    return K.variable(value=data)


def test_bottleneck():
    input_tensor = generate_tensor(input_shape=(4, 16, 16, 128))
    output_tensor = generate_tensor(input_shape=(4, 16, 16, 16))
    assert np.equal(sut.bottleneck(input_tensor, 16).get_shape(), output_tensor.get_shape())

    output_tensor = generate_tensor(input_shape=(4, 16, 16, 8))
    assert np.equal(sut.bottleneck(input_tensor, 8).get_shape(), output_tensor.get_shape())


# WIP test
# def test_bottleneck_eval():
#     with K.get_session() as sess:
#         input_tensor = K.constant(generate_data(input_shape=(4, 16, 16, 128), fill_value=0))
#         output_tensor = K.constant(generate_data(input_shape=(4, 16, 16, 16), fill_value=0))
#         output_tensor = PReLU(shared_axes=[1, 2])(output_tensor)
#         assert sut.bottleneck(input_tensor, 16).eval(session=sess) == output_tensor.eval(session=sess)
