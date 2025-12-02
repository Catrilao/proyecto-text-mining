import time
import numpy as np
import collections
import random

from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Activation,
    LSTM,
    GRU,
    Bidirectional,
    Conv1D,
    Flatten,
    MaxPooling1D,
)


class TweetModels:
    ###################################################################################################
    def create_model_LSTM(
        self, _tree_max_num_seq, _emb_size, _num_categories, _units=200, _dropout=0.3
    ):
        model = Sequential()
        model.add(
            LSTM(
                _units,
                input_shape=(_tree_max_num_seq, _emb_size),
                return_sequences=False,
            )
        )
        model.add(Dropout(_dropout))
        model.add(Dense(_num_categories))
        model.add(Activation("softmax"))
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    ###################################################################################################
    def create_model_StackedLSTM(
        self, _tree_max_num_seq, _emb_size, _num_categories, _units=200, _dropout=0.3
    ):
        model = Sequential()
        model.add(
            LSTM(
                _units,
                input_shape=(_tree_max_num_seq, _emb_size),
                return_sequences=True,
            )
        )
        model.add(LSTM(_units, return_sequences=False))
        model.add(Dropout(_dropout))
        model.add(Dense(_num_categories))
        model.add(Activation("softmax"))
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    #! USAR ESTA
    ###################################################################################################
    def create_model_GRU(
        self, _tree_max_num_seq, _emb_size, _num_categories, _units=200, _dropout=0.3
    ):
        model = Sequential()
        model.add(
            GRU(
                _units,
                input_shape=(_tree_max_num_seq, _emb_size),
                return_sequences=False,
            )
        )
        model.add(Dropout(_dropout))
        model.add(Dense(_num_categories))
        model.add(Activation("softmax"))
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    ###################################################################################################
    def create_model_StackedGRU(
        self, _tree_max_num_seq, _emb_size, _num_categories, _units=200, _dropout=0.3
    ):
        model = Sequential()
        model.add(
            GRU(
                _units,
                input_shape=(_tree_max_num_seq, _emb_size),
                return_sequences=True,
            )
        )
        model.add(GRU(_units, return_sequences=False))
        model.add(Dropout(_dropout))
        model.add(Dense(_num_categories))
        model.add(Activation("softmax"))
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    ###################################################################################################
    def create_model_BI_LSTM(
        self, _tree_max_num_seq, _emb_size, _num_categories, _units=200, _dropout=0.3
    ):
        model = Sequential()
        model.add(
            Bidirectional(
                LSTM(
                    _units,
                    input_shape=(_tree_max_num_seq, _emb_size),
                    return_sequences=False,
                )
            )
        )
        model.add(Dropout(_dropout))
        model.add(Dense(_num_categories))
        model.add(Activation("softmax"))
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    ###################################################################################################
    def create_model_BI_StackedLSTM(
        self, _tree_max_num_seq, _emb_size, _num_categories, _units=200, _dropout=0.3
    ):
        model = Sequential()
        model.add(
            Bidirectional(
                LSTM(
                    _units,
                    input_shape=(_tree_max_num_seq, _emb_size),
                    return_sequences=True,
                )
            )
        )
        model.add(Bidirectional(LSTM(_units, return_sequences=False)))
        model.add(Dropout(_dropout))
        model.add(Dense(_num_categories))
        model.add(Activation("softmax"))
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    ###################################################################################################
    def create_model_BI_GRU(
        self, _tree_max_num_seq, _emb_size, _num_categories, _units=200, _dropout=0.3
    ):
        model = Sequential()
        model.add(
            Bidirectional(
                GRU(
                    _units,
                    input_shape=(_tree_max_num_seq, _emb_size),
                    return_sequences=False,
                )
            )
        )
        model.add(Dropout(_dropout))
        model.add(Dense(_num_categories))
        model.add(Activation("softmax"))
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    ###################################################################################################
    def create_model_BI_StackedGRU(
        self, _tree_max_num_seq, _emb_size, _num_categories, _units=200, _dropout=0.3
    ):
        model = Sequential()
        model.add(
            Bidirectional(
                GRU(
                    _units,
                    input_shape=(_tree_max_num_seq, _emb_size),
                    return_sequences=True,
                )
            )
        )
        model.add(Bidirectional(GRU(_units, return_sequences=False)))
        model.add(Dropout(_dropout))
        model.add(Dense(_num_categories))
        model.add(Activation("softmax"))
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    ###################################################################################################
    def create_model_Conv1D(
        self,
        _tree_max_num_seq,
        _emb_size,
        _num_categories,
        _units=200,
        _dropout=0.3,
        _kernel_size=2,
    ):
        model = Sequential()
        model.add(
            Conv1D(
                _units,
                _kernel_size,
                activation="relu",
                input_shape=(_tree_max_num_seq, _emb_size),
            )
        )
        model.add(MaxPooling1D())
        model.add(Conv1D(_units, _kernel_size, activation="relu"))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dropout(_dropout))
        model.add(Dense(_num_categories))
        model.add(Activation("softmax"))
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    ###################################################################################################
    def create_model_RCNN(
        self,
        _tree_max_num_seq,
        _emb_size,
        _num_categories,
        _units=200,
        _dropout=0.3,
        _kernel_size=2,
    ):
        model = Sequential()
        model.add(
            Conv1D(
                _units,
                _kernel_size,
                activation="relu",
                input_shape=(_tree_max_num_seq, _emb_size),
            )
        )
        model.add(MaxPooling1D())
        model.add(Conv1D(_units, _kernel_size, activation="relu"))
        model.add(MaxPooling1D())
        model.add(LSTM(_units, return_sequences=True, recurrent_dropout=_dropout))
        model.add(LSTM(_units, recurrent_dropout=_dropout))
        model.add(Dense(1024, activation="relu"))
        model.add(Dense(_num_categories, activation="softmax"))
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    ###################################################################################################
