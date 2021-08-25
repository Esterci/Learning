from os import times
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,normalize, MinMaxScaler
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
from keras.layers import Dense, Input, Concatenate, Flatten, BatchNormalization, Dropout, LeakyReLU
from keras.models import Sequential, Model
from keras.backend import clear_session
from keras.losses import binary_crossentropy
from Disco_tensor_flow import distance_corr
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
from datetime import datetime, time

def time_stamp():
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H.%M.%S")
    return timestampStr

# define new loss with distance decorrelation
def decorr(var_1, var_2, weights,kappa):

    def loss(y_true, y_pred):
        #return binary_crossentropy(y_true, y_pred) + distance_corr(var_1, var_2, weights)
        #return distance_corr(var_1, var_2, weights)
        return binary_crossentropy(y_true, y_pred) + kappa * distance_corr(var_1, var_2, weights,power=2)
        #return binary_crossentropy(y_true, y_pred)

    return loss


#########################################################
# ------------------------------------------------------ #
# ----------------------- LOADING ---------------------- #
# ------------------------------------------------------ #
##########################################################
# Firstly the model loads the background and signal data, 
# then it removes the attributes first string line, which 
# are the column names, in order to avoid NaN values in 
# the array.

print('==== Commencing Initiation ====\n')

### Background
b_name='/home/thiago/Documents/Data_Sets/LPC-anomaly-detection/Input_Background_1.csv'
background = np.genfromtxt(b_name, delimiter=',')
background = background[1:,:]
print(".Background Loaded..." )
print(".Background shape: {}".format(background.shape))

### Signal
s_name='/home/thiago/Documents/Data_Sets/LPC-anomaly-detection/Input_Signal_1.csv'
signal = np.genfromtxt(s_name, delimiter=',')
signal = signal[1:,:]
print(".Signal Loaded...")
print(".Signal shape: {}\n".format(signal.shape))

##########################################################
# ------------------------------------------------------ #
# --------------------- INITIATION --------------------- #
# ------------------------------------------------------ #
##########################################################

# Number of events
total = 100000

# Percentage of background samples on the testing phase
background_percent = 0.99

# Percentage of samples on the training phase
test_size = 0.3

# Number of iterations

n_it = 33

# Defining hyper-parameters range

min_batch_size = 200

max_batch_size = 500

min_hidden_dim = 8

max_hidden_dim = 21

print('\n          ==== Initiation Complete ====\n')
print('=*='*17 )
print('\n      ==== Commencing Pre-processing ====\n')

# Percentage of background samples to divide the data-set
dat_set_percent = total/len(background)

# Reducing background samples
_,reduced_background = train_test_split(background, test_size=dat_set_percent)

# Deviding train and test background

train_data, background_test = train_test_split(reduced_background, test_size=test_size)

# Iserting the correct number of signal in streaming

n_signal_samples = int(len(background_test)*(1-background_percent))

_,background_test = train_test_split(background_test, test_size=background_percent)

_,signal_test = train_test_split(signal, test_size=n_signal_samples/len(signal))

# Concatenating Signal and the Background sub-sets

test_data = np.vstack((background_test,signal_test))

# Normalize Data

print('.Normalizing Data')

test_data = normalize(test_data,
                        norm='max',
                        axis=0
                    )

train_data = normalize(train_data,
                        norm='max',
                        axis=0
                    )

# Creates test data frame

attributes = np.array(["px1",
                        "py1",
                        "pz1",
                        "E1",
                        "eta1",
                        "phi1",
                        "pt1",
                        "px2",
                        "py2",
                        "pz2",
                        "E2",
                        "eta2",
                        "phi2",
                        "pt2",
                        "Delta_R",
                        "M12",
                        "MET",
                        "S",
                        "C",
                        "HT",
                        "A"]
                      )

test_df = pd.DataFrame(test_data,columns = attributes)

# Creating Labels
print('.Creating Labels')

test_labels =np.ones((len(test_data)))
test_labels[:len(background_test)] = 0

print('\n      ==== Pre-processing Complete ====\n')
print(".Train data shape: {}".format(train_data.shape))
print(".Test data shape: {}".format(test_data.shape))
print(".Test Background shape: {}".format(background_test.shape))
print(".Test Signal shape: {}".format(signal_test.shape))

print('=*='*17 )

# Creating log file 

with open('log_file.txt', 'w') as f:
    f.write('======= New Analysis ' + time_stamp() + ' ======\n')
    f.write('\n.Train data shape: {}'.format(train_data.shape))
    f.write('\n.Test data shape: {}'.format(test_data.shape))
    f.write('\n.Test Background shape: {}'.format(background_test.shape))
    f.write('\n.Test Signal shape: {}'.format(signal_test.shape))
    f.write('\n.Number of events: {}'.format(total))
    f.write('\n.Percentage of background samples on the testing phase: {}'.format(background_percent))
    f.write('\n.Percentage of samples on the training phase: {}'.format(test_size))
    f.write('\n.Number of iterations: {}'.format(n_it))
    f.write('\n.Defining hyper-parameters ranges')
    f.write('    \n.min_batch_size: {}'.format(min_batch_size))
    f.write('    \n.max_batch_size: {}'.format(max_batch_size))
    f.write('    \n.min_hidden_dim: {}'.format(min_hidden_dim))
    f.write('    \n.max_hidden_dim: {}\n'.format(max_hidden_dim))

# Parameters in study

batch_size_list = list(np.linspace(min_batch_size,max_batch_size,num=3,dtype=int))
encoding_dim_list = list(np.linspace(max_hidden_dim,min_hidden_dim,num=3,dtype=int))
lambda_disco_list = list(np.linspace(0,600,num=3,dtype=int))
act_func_list_1 = ['relu',
                   'sigmoid',
                   #'softmax',
                   #'softplus',
                   #'softsign',
                   'tanh',
                   #'selu',
                   #'elu',
                   #'exponential'
                   ]
act_func_list_2 = act_func_list_1
act_func_list_3 = act_func_list_1

n_combinations = (len(batch_size_list) * 
                    len(encoding_dim_list) * 
                    len(lambda_disco_list) * 
                    len(act_func_list_1) * 
                    len(act_func_list_1) * 
                    len(act_func_list_1)
                )

for it in range(n_it):

    combinations_count = 0

    with open('log_file.txt', 'a') as f:
        f.writelines('\nIteration number ' + str(it) + ' ==> ' + time_stamp())

    for batch_size in batch_size_list:
        for encoding_dim in encoding_dim_list:
            for lambda_disco in lambda_disco_list:
                for act_1 in act_func_list_1:
                    for act_2 in act_func_list_2:
                        for act_3 in act_func_list_3:

                            # Fixed parameters

                            nb_epoch = 100
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
                                'it' + '__' + str(it+1) + '__' + time_stamp()
                            )

                            ###### Creates NN structure #####
                            
                            #input Layer
                            input_layer = Input(shape=(input_dim, ))
                            sample_weights = Input(shape=(1, ))
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
                            
                            opt = Adam(lr=learning_rate)
                            autoencoder.compile(optimizer=opt, 
                                                loss=decorr(input_layer[:,15], 
                                                            decoder[:,15], 
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
                                
                                test_x_predictions = autoencoder.predict([test_data,
                                                                        np.ones(len(test_data))]
                                                                        )
                                # Calculating MSE
                    
                                mse = np.mean(np.power(test_data - test_x_predictions, 2), 
                                            axis=1
                                            )
                        
                                # Creating MSE data frame
                        
                                error_df = pd.DataFrame({'Reconstruction_error': mse,
                                                        'Class': test_labels
                                                        }
                                                    )
                                # Concatenating with the original Attributes
                                
                                results_df = pd.concat([test_df,error_df],
                                                    axis=1
                                                    )
                                
                                # Saving Results
                                
                                results_df.to_csv('Results/results__' + struct_name + '__.csv')

                                # Ploting Reconstitution error
                                
                                groups = error_df.groupby('Class')
                                fig, ax = plt.subplots()
                                for name, group in groups:
                                    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
                                            label= "Signal" if name == 1 else "Background")
                                ax.legend()
                                plt.title("Reconstruction error")
                                plt.ylabel("Reconstruction error")
                                plt.xlabel("Data point index")
                                
                                fig.savefig('Figures/reconstruction-error__' + struct_name + '__.png', 
                                            bbox_inches='tight'
                                        )

                            except:
                                results_df.to_csv('Results/results__' + struct_name + '__.csv')
                                with open('Error__' + struct_name + '.txt', 'w') as f:
                                    f.write('error')

                            combinations_count += 1

                            with open('log_file.txt', 'a') as f:
                                f.writelines('\n    .{} of {} combinations at '.format(combinations_count,n_combinations) + time_stamp())

with open('log_file.txt', 'a') as f:
    f.writelines('\n======== Analysis Complete ' + time_stamp() + ' =========')
