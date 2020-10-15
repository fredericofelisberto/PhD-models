import numpy as np
from keras.layers import LSTM, Flatten, Dropout, Dense, TimeDistributed
from keras.models import Sequential
from sklearn.utils.class_weight import compute_class_weight


def RNN(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(32, activation='relu')))
    model.add(TimeDistributed(Dense(16, activation='relu')))
    model.add(TimeDistributed(Dense(8, activation='relu')))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.add(Flatten())
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model


def compute_weight(y):
    y_flat = np.argmax(y, axis=1)
    weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)
    return weight


def fit(model, X, y, epochs):
    model.fit(X, y, epochs, verbose=1, class_weight=compute_weight(y))
