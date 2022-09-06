import numpy as np
import os

from tensorflow.keras.layers import Input, Dense, Embedding, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Dropout, concatenate, Activation, SpatialDropout1D
from tensorflow.keras.layers import Reshape, Flatten
from tensorflow.keras.models import Model

from statics import *

def cnn_model(input_dim, output_shape, path=''):
    embeddings_path = os.path.join(path, 'embeddings.npy')
    weights = np.load(embeddings_path)
    embedding_dim = weights.shape[1]

    inputs = Input(shape=(input_dim,), dtype='int32')

    embedding = Embedding(output_dim=weights.shape[1], input_dim=weights.shape[0], input_length=input_dim,
                          weights=[weights], trainable=True)(inputs)

    spatial_dropout = SpatialDropout1D(0.5)(embedding)

    reshape = Reshape((input_dim, embedding_dim, 1))(spatial_dropout)

    conv_0 = Conv2D(num_filters, (filter_sizes[0], embedding_dim), padding='valid', kernel_initializer='normal',
                    activation='sigmoid', data_format='channels_last')(reshape)
    conv_1 = Conv2D(num_filters, (filter_sizes[1], embedding_dim), padding='valid', kernel_initializer='normal',
                    activation='sigmoid', data_format='channels_last')(reshape)
    conv_2 = Conv2D(num_filters, (filter_sizes[2], embedding_dim), padding='valid', kernel_initializer='normal',
                    activation='sigmoid', data_format='channels_last')(reshape)
    conv_3 = Conv2D(num_filters, (filter_sizes[3], embedding_dim), padding='valid', kernel_initializer='normal',
                    activation='sigmoid', data_format='channels_last')(reshape)

    maxpool_0 = MaxPooling2D(pool_size=(input_dim - filter_sizes[0] + 1, 1), strides=(1, 1),
                             padding='valid', data_format='channels_last')(conv_0)
    maxpool_1 = MaxPooling2D(pool_size=(input_dim - filter_sizes[1] + 1, 1), strides=(1, 1),
                             padding='valid', data_format='channels_last')(conv_1)
    maxpool_2 = MaxPooling2D(pool_size=(input_dim - filter_sizes[2] + 1, 1), strides=(1, 1),
                             padding='valid', data_format='channels_last')(conv_2)
    maxpool_3 = MaxPooling2D(pool_size=(input_dim - filter_sizes[3] + 1, 1), strides=(1, 1),
                             padding='valid', data_format='channels_last')(conv_3)

    merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2, maxpool_3], axis=1)

    flatten = Flatten()(merged_tensor)

    # dense1 = Dense(units=output_dim, kernel_regularizer=regularizers.l2(0.01))(flatten)
    dense1 = Dense(units=output_dim)(flatten)
    # dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense1 = Dropout(drop)(dense1)

    # dense2 = Dense(units=output_dim, kernel_regularizer=regularizers.l2(0.01))(dense1)
    dense2 = Dense(units=output_dim)(dense1)
    # dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    dense2 = Dropout(drop)(dense2)

    # dense2 = Dense(units=output_dim, activation='relu')(dense1)
    # dense3 = Dense(units=output_dim, activation='relu')(dense2)
    output = Dense(units=output_shape)(dense1)
    # output = BatchNormalization()(output)
    output = Activation('softmax')(output)

    # output = Dense(units=output_shape, activation='softmax')(normalized_1)

    model = Model(inputs=inputs, outputs=output)

    return model