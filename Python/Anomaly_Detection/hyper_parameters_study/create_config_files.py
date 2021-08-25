import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, Flatten

from saphyra import *

def get_model():
    modelCol = []
    for n in range(10, 15):
        model = Sequential()
        model.add(Dense(n, input_shape=(100,), activation='tanh', name='dense_layer'))
        model.add(Dense(1, activation='linear', name='output_for_inference'))
        model.add(Activation('tanh', name='output_for_training'))
        modelCol.append(model)
    
    return modelCol


create_jobs(models = get_model(),
            nInits = 1,
            nInitsPerJob = 1,
            sortBounds = 10,
            nSortsPerJob = 1,
            nModelsPerJob = 1,
            outputFolder = 'job_test2')