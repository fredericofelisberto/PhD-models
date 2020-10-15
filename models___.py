import numpy as np
from keras.layers import LSTM, Flatten, Dropout, Dense, TimeDistributed
from keras.models import Sequential
from sklearn.utils.class_weight import compute_class_weight


class RNN:
    def __init__(self,  X, y, opt, input_shape):
        super(RNN, self).__init__()

        self.opt = opt
        self.X = X
        self.y = y
        self.epochs = opt.epochs

        self.model = Sequential()
        self.model .add(LSTM(128, return_sequences=True, input_shape=input_shape))
        self.model .add(LSTM(128, return_sequences=True))
        self.model .add(Dropout(0.5))
        self.model .add(TimeDistributed(Dense(32, activation='relu')))
        self.model .add(TimeDistributed(Dense(16, activation='relu')))
        self.model .add(TimeDistributed(Dense(8, activation='relu')))
        self.model .add(Flatten())
        self.model .add(Dense(10, activation='softmax'))
        self.model .add(Flatten())
        self.model .summary()
        self.model .compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    def compute_weight(self):
        y_flat = np.argmax(self.y, axis=1)
        weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)
        return weight

    def fit_(self):
        self.model.fit(self.X, self.y, self.opt.epochs, class_weight=self.compute_weight())
