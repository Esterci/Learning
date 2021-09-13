from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from Disco_tensor_flow import decorr
from keras.optimizers import Adam
from datetime import datetime
import argparse
import pickle as pk



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

parser.add_argument('-l','--lambda_disco', action='store',
        dest='lambda_disco', required = True,
            help = "The job config file that will be used to configure the job (sort and init).")

parser.add_argument('-a1','--act_1', action='store',
        dest='act_1', required = False, default = None,
            help = "The volume output.")

parser.add_argument('-a2','--act_2', action='store',
        dest='act_2', required = False, default = None,
            help = "The volume output.")

parser.add_argument('-a3','--act_3', action='store',
        dest='act_3', required = False, default = None,
            help = "The volume output.")

parser.add_argument('-dcorr','--dcorr', action='store',
        dest='dcorr', required = False, default = None,
            help = "The volume output.")

parser.add_argument('-f','--file', action='store',
        dest='file', required = False, default = None,
            help = "The volume output.")

args = parser.parse_args()

batch_size = int(args.batch_size)
encoding_dim = int(args.encoding_dim)
lambda_disco = int(args.lambda_disco)
dcorr = int(args.dcorr)
act_1 = args.act_1
act_2 = args.act_2
act_3 = args.act_3
file = args.file

it = file.split('__')[8]

print(it)

s = file

with open(s, 'rb') as f:
    data_dict = pk.load(f)

train_data = data_dict['train_data']
test_labels = data_dict['test_labels']
test_data = data_dict['test_data']
test_df = data_dict['test_df']

# Fixed parameters

nb_epoch = 150
input_dim = train_data.shape[1]
hidden_dim_1 = int(encoding_dim / 2)
hidden_dim_2 = int(hidden_dim_1 / 2)
learning_rate = 0.001

##### Creates structure name #####

struct_name = (
    'batch_size' + '__' + str(batch_size) + '__' +
    'encoding_dim' + '__' + str(encoding_dim) + '__' +
    'lambda_disco' + '__' + str(lambda_disco) + '__' +
    'act_1' + '__' + str(act_1) + '__' +
    'act_2' + '__' + str(act_2) + '__' +
    'act_3' + '__' + str(act_3) + '__' +
    'it' + '__' + it + '__' + time_stamp()
)

###### Creates NN structure #####

#input Layer
input_layer = tf.keras.layers.Input(shape=(input_dim, ))
sample_weights = tf.keras.layers.Input(shape=(1, ))
#Encoder
encoder = tf.keras.layers.Dense(encoding_dim, 
                                activation=act_1,
        activity_regularizer=tf.keras.regularizers.l2(learning_rate)
                            )(input_layer)

encoder = tf.keras.layers.Dropout(0.2)(encoder)

encoder = tf.keras.layers.Dense(hidden_dim_1, 
                                activation=act_2
                            )(encoder)

encoder = tf.keras.layers.Dense(hidden_dim_2, 
                                activation=act_3
                            )(encoder)
# Decoder
decoder = tf.keras.layers.Dense(hidden_dim_1, 
                                activation=act_3
                            )(encoder)

decoder=tf.keras.layers.Dropout(0.2)(decoder)

decoder = tf.keras.layers.Dense(encoding_dim, 
                                activation=act_2
                            )(decoder)

decoder = tf.keras.layers.Dense(input_dim, 
                                activation=act_1
                            )(decoder)
#Autoencoder
autoencoder = tf.keras.Model(inputs=[input_layer, sample_weights], 
                            outputs=decoder
                            )

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
                    loss=decorr(input_layer[:,dcorr], 
                                decoder[:,dcorr], 
                                sample_weights[:,0],
                                lambda_disco)
                )

# Training
try:
    Train = [train_data, np.ones(len(train_data))]

    history = autoencoder.fit(x = Train, 
                            y = train_data,
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

    test_x_predictions = autoencoder.predict([test_data,
                                            np.ones(len(test_data))]
                                            )
    # Calculating MSE

    end = datetime.now()

    error = np.power(np.power(test_data - test_x_predictions, 2),0.5)

    # Creating MSE data frame
    columns = ["px1","py1","pz1","E1","eta1",
                "phi1","pt1","px2","py2","pz2",
                "E2","eta2","phi2","pt2","Delta_R",
                "M12","MET","S","C","HT","A"]
    error_df = pd.DataFrame(error,columns=columns)

    id_df = pd.DataFrame({'class': test_labels,
                          'time' : end -start
                            }
                        )

    # Creating 
    # Concatenating with the original Attributes
    
    results_df = pd.concat([test_df,error_df],
                        axis=1
                        )
    
    # Saving Results
    
    results_df.to_csv('Results/results__' + struct_name + '__.csv')

    # Ploting Reconstitution error
    
    groups = error_df.groupby('class')
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',
                label= "Signal" if name == 1 else "Background")
    ax.legend()
    plt.title("Reconstruction error")
    plt.ylabel("Reconstruction error")
    plt.xlabel("Data point index")
    
    fig.savefig('Figures/reconstruction-error__' + struct_name + '__.png', 
                bbox_inches='tight'
            )

except:
    with open('Results/Error__' + struct_name + '.txt', 'w') as f:
        f.write('error')

# Clear everything for the next init
K.clear_session()        
