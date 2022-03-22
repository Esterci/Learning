from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from keras.optimizers import Adam
from keras.layers import Dense, Input, BatchNormalization, Dropout, LeakyReLU
from keras.models import Model
from keras.losses import binary_crossentropy
from datetime import datetime
import argparse
import pickle as pk

def get_block(L, size):
    L = BatchNormalization()(L)

    L = Dense(size)(L)
    L = Dropout(0.5)(L)
    L = LeakyReLU(0.2)(L)
    return L


def time_stamp():
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H.%M.%S")
    return timestampStr



parser = argparse.ArgumentParser(description = '', add_help = False)
parser = argparse.ArgumentParser()


parser.add_argument('-b','--batch_size', action='store',
        dest='batch_size', required = True,
            help = "The job config file that will be used to configure the job (sort and init).")

parser.add_argument('-e','--encoding_dim', action='store',
        dest='encoding_dim', required = False, default = None,
            help = "The volume output.")

parser.add_argument('-f','--file', action='store',
        dest='file', required = False, default = None,
            help = "The volume output.")

args = parser.parse_args()

batch_size = int(args.batch_size)
encoding_dim = int(args.encoding_dim)
file = args.file

it = file.split('__')[8]

print(it)

s = file

with open(s, 'rb') as f:
    data_dict = pk.load(f)

train_data = data_dict['train_data']
train_data = train_data

test_labels = data_dict['test_labels']
test_data = data_dict['test_data']
test_df = data_dict['test_df']

train_jets = train_data[:,:14]
train_colision = train_data[:,14:]

test_jets = test_data[:,:14]
test_colision = test_data[:,14:]


# Fixed parameters

nb_epoch = 150
input_dim = train_jets.shape[1]
hidden_dim_1 = int(encoding_dim / 2)
hidden_dim_2 = int(hidden_dim_1 / 2)
learning_rate = 0.001   

##### Creates structure name #####

struct_name = (
    'batch_size' + '__' + str(batch_size) + '__' +
    'encoding_dim' + '__' + str(encoding_dim) + '__' +
    'it' + '__' + it + '__' + time_stamp()
)

###### Creates NN structure #####

# Setup network
# make inputs

jets = Input(shape=input_dim)
sample_weights = Input(shape=(1,))

#setup trainable layers
d1 = get_block(jets, encoding_dim)
d2 = get_block(d1, hidden_dim_1)
d3 = get_block(d2, hidden_dim_2)
d4 = get_block(d3, hidden_dim_2)
d5 = get_block(d4, hidden_dim_1)
o = get_block(d5, 7)

autoencoder = Model(inputs=[jets, sample_weights], outputs=o)

autoencoder.summary()

#Defining early stop

cp = tf.keras.callbacks.ModelCheckpoint(filepath="autoencoder.h5",
                                        mode='min', 
                                        monitor='val_loss', 
                                        verbose=2, 
                                        save_best_only=True
                                    )

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=20,
    verbose=1, 
    mode='min',
    restore_best_weights=True
)

# Compiling NN

opt = Adam(lr=0.001)
autoencoder.compile(optimizer=opt, 
                    loss='binary_crossentropy'
                )
                

# Training
#try:
Train = [train_jets, np.ones(len(train_jets))]

history = autoencoder.fit(x = Train, 
                        y = train_colision,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=0.1,
                        verbose=1,
                        callbacks=[cp, early_stop]
                        ).history

# Ploting Model Loss

fig, ax = plt.subplots()
plt.plot(history['loss'], linewidth=2, label='Train')
plt.plot(history['val_loss'], linewidth=2, label='Test')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

fig.savefig('Figures/model-loss__' + struct_name + '__.png', 
            bbox_inches='tight'
        )

# Predicting Test values

start = datetime.now()

test_x_predictions = autoencoder.predict([test_jets,
                                        np.ones(len(test_jets))]
                                        )
# Calculating MSE

end = datetime.now()

error = np.power(np.power(test_colision - test_x_predictions, 2),0.5)

# Creating MSE data frame

columns = ["Delta_R","M12","MET","S","C","HT","A"]

error_df = pd.DataFrame(error,columns=columns)

id_df = pd.DataFrame({'class': test_labels,
                        'time' : end - start
                        }
                    )

# Concatenating with the original Attributes

results_df = pd.concat([error_df,id_df],
                    axis=1
                    )

# Covnert pandas data-frame to array

results = results_df.values

# Saving Results

np.save('Results/results__' + struct_name + '__ ', results)

#except:
#    with open('Results/Error__' + struct_name + '.txt', 'w') as f:
#        f.write('error')

# Clear everything for the next init
K.clear_session()        
