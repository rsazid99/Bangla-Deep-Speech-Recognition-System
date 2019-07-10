from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input,
                          TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)

import numpy as np

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers.recurrent import SimpleRNN
from keras.layers import Dense, Activation, Bidirectional, Reshape, Flatten, Lambda, Input,\
    Masking, Convolution1D, BatchNormalization, GRU, Conv1D, RepeatVector, Conv2D
from keras.optimizers import SGD, adam
from keras.layers import ZeroPadding1D, Convolution1D, ZeroPadding2D, Convolution2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import TimeDistributed, Dropout
from keras.layers.merge import add  # , # concatenate BAD FOR COREML
from keras.utils.conv_utils import conv_output_length
from keras.activations import relu

import tensorflow as tf


def clipped_relu(x):
    return relu(x, max_value=20)

# Define CTC loss


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    # hack for load_model
    import tensorflow as tf

    ''' from TF: Input requirements
    1. sequence_length(b) <= time for all b
    2. max(labels.indices(labels.indices[:, 1] == b, 2)) <= sequence_length(b) for all b.
    '''

    # print("CTC lambda inputs / shape")
    # print("y_pred:",y_pred.shape)  # (?, 778, 30)
    # print("labels:",labels.shape)  # (?, 80)
    # print("input_length:",input_length.shape)  # (?, 1)
    # print("label_length:",label_length.shape)  # (?, 1)

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def ctc(y_true, y_pred):
    return y_pred


def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True,
                   implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = LSTM(units, activation=activation,
                    return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn_1d')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
                  conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu',
                   return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn_1d')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model


def cnn_output_length(input_length, filter_size, border_mode, stride,
                      dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride


def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    if recur_layers == 1:
        layer = LSTM(units, return_sequences=True, activation='relu')(input_data)
        layer = BatchNormalization(name='bt_rnn_1')(layer)
    else:
        layer = LSTM(units, return_sequences=True, activation='relu')(input_data)
        layer = BatchNormalization(name='bt_rnn_1')(layer)

        for i in range(recur_layers - 2):
            layer = LSTM(units, return_sequences=True, activation='relu')(layer)
            layer = BatchNormalization(name='bt_rnn_{}'.format(2 + i))(layer)

        layer = LSTM(units, return_sequences=True, activation='relu')(layer)
        layer = BatchNormalization(name='bt_rnn_last_rnn')(layer)

    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(layer)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(LSTM(units, return_sequences=True, activation='relu'), merge_mode='concat')(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def final_model(input_dim, filters, kernel_size, conv_stride,
                conv_border_mode, units, output_dim=29, dropout_rate=0.5, number_of_layers=3,
                cell=GRU, activation='tanh'):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='layer_1_conv',
                     dilation_rate=1)(input_data)
    conv_bn = BatchNormalization(name='conv_batch_norm')(conv_1d)

    if number_of_layers == 1:
        layer = cell(units, activation=activation,
                     return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate)(conv_bn)
        layer = BatchNormalization(name='bt_rnn_1')(layer)
    else:
        layer = cell(units, activation=activation,
                     return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate)(conv_bn)
        layer = BatchNormalization(name='bt_rnn_1')(layer)

        for i in range(number_of_layers - 2):
            layer = cell(units, activation=activation,
                         return_sequences=True, implementation=2, name='rnn_{}'.format(i + 2), dropout=dropout_rate)(layer)
            layer = BatchNormalization(name='bt_rnn_{}'.format(i + 2))(layer)

        layer = cell(units, activation=activation,
                     return_sequences=True, implementation=2, name='final_layer_of_rnn')(layer)
        layer = BatchNormalization(name='bt_rnn_final')(layer)

    time_dense = TimeDistributed(Dense(output_dim))(layer)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model


def ds2(input_dim=161, fc_size=512, rnn_size=512, output_dim=83, initialization='glorot_uniform',
        conv_layers=1, gru_layers=1, use_conv=True):
    """ DeepSpeech 2 implementation
    Architecture:
        Input Spectrogram TIMEx161
        1 Batch Normalisation layer on input
        1-3 Convolutional Layers
        1 Batch Normalisation layer
        1-7 BiDirectional GRU Layers
        1 Batch Normalisation layer
        1 Fully connected Dense
        1 Softmax output
    Details:
       - Uses Spectrogram as input rather than MFCC
       - Did not use BN on the first input
       - Network does not dynamically adapt to maximum audio size in the first convolutional layer. Max conv
          length padded at 2048 chars, otherwise use_conv=False
    Reference:
        https://arxiv.org/abs/1512.02595
    """

    K.set_learning_phase(1)

    input_data = Input(shape=(None, input_dim), name='the_input')
    bn_1 = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(input_data)

#     if use_conv:
#         conv = ZeroPadding1D(padding=(0, 2048))(x1)
#         for l in range(conv_layers):
#             x1 = Conv1D(filters=fc_size, name='conv_{}'.format(l+1), kernel_size=11, padding='valid', activation='relu', strides=2)(conv)
#     else:
#         for l in range(conv_layers):
#             x1 = TimeDistributed(Dense(fc_size, name='fc_{}'.format(l + 1), activation='relu'))(x1)  # >>(?, time, fc_size)

    conv = ZeroPadding1D(padding=(0, 2048))(bn_1)
    conv_1 = Conv1D(filters=1280, name='conv_{}'.format(1), kernel_size=11, padding='valid', activation='relu', strides=2)(conv)

    #conv_2 = Conv1D(filters=640, name='conv_{}'.format(2), kernel_size=5, padding='valid', activation='relu', strides=2)(conv_1)
    #conv_3 = Conv1D(filters=512, name='conv_{}'.format(3), kernel_size=5, padding='valid', activation='relu', strides=2)(conv_2)

    bn_2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(conv_1)

#     for l in range(gru_layers):
#         x1 = Bidirectional(GRU(rnn_size, name='fc_{}'.format(l + 1), return_sequences=True, activation='relu', kernel_initializer=initialization),
#                       merge_mode='sum')(x1)
    gru_1 = Bidirectional(GRU(rnn_size, name='fc_{}'.format(1), return_sequences=True, activation='relu', kernel_initializer=initialization), merge_mode='sum')(bn_2)
#     gru_2 = Bidirectional(GRU(rnn_size, name='fc_{}'.format(2), return_sequences=True, activation='relu', kernel_initializer=initialization), merge_mode='sum')(gru_1)
#     gru_3 = Bidirectional(GRU(rnn_size, name='fc_{}'.format(3), return_sequences=True, activation='relu', kernel_initializer=initialization), merge_mode='sum')(gru_2)

    bn_3 = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(gru_1)

    # Last Layer 5+6 Time Dist Dense Layer & Softmax
    tm_1 = TimeDistributed(Dense(256, activation=clipped_relu))(bn_3)
    y_pred = TimeDistributed(Dense(output_dim, name="y_pred", activation="softmax"))(tm_1)

#     # labels = K.placeholder(name='the_labels', ndim=1, dtype='int32')
#     labels = Input(name='the_labels', shape=[None,], dtype='int32')
#     input_length = Input(name='input_length', shape=[1], dtype='int32')
#     label_length = Input(name='label_length', shape=[1], dtype='int32')

#     # Keras doesn't currently support loss funcs with extra parameters
#     # so CTC loss is implemented in a lambda layer
#     loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,
#                                                                        labels,
#                                                                        input_length,
#                                                                        label_length])

#     model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())

    return model
