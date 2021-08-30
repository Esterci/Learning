import tensorflow as tf
import numpy as np
from saphyra import *

try:
  from tensorflow.compat.v1 import ConfigProto
  from tensorflow.compat.v1 import InteractiveSession

  config = ConfigProto()
  config.gpu_options.allow_growth = True
  session = InteractiveSession(config=config)
except Exception as e:
  print(e)
  print("Not possible to set gpu allow growth")



##########################################################
# ------------------------------------------------------ #
# --------------------- INITIATION --------------------- #
# ------------------------------------------------------ #
##########################################################

# Defining hyper-parameters range

min_batch_size = 200

max_batch_size = 500

min_hidden_dim = 8

max_hidden_dim = 21

batch_size_list = list(np.linspace(min_batch_size,max_batch_size,num=3,dtype=int))

lambda_disco_list = list(np.linspace(0,600,num=3,dtype=int))

def get_model(min_hidden_dim,max_hidden_dim):
    # Parameters in study

    encoding_dim_list = list(np.linspace(max_hidden_dim,min_hidden_dim,num=3,dtype=int))
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

    modelCol = []

    for encoding_dim in encoding_dim_list:
        for act_1 in act_func_list_1:
            for act_2 in act_func_list_2:
                for act_3 in act_func_list_3:

                    # Fixed parameters

                    nb_epoch = 100
                    input_dim = 21
                    hidden_dim_1 = int(encoding_dim / 2)
                    hidden_dim_2 = int(hidden_dim_1 / 2)
                    learning_rate = 0.001

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
                    
                    modelCol.append(autoencoder)

    return  modelCol #lista de NN keras

for batch in batch_size_list:
    for l_disco in lambda_disco_list:
        create_jobs(models        = get_model(min_hidden_dim,
                                            max_hidden_dim),
                    epochs        = 150,
                    batch_size    = batch,
                    lambda_disco  = l_disco,
                    nInits        = 1,
                    nInitsPerJob  = 1,
                    sortBounds    = 1,
                    nSortsPerJob  = 1,
                    nModelsPerJob = 6,
                    outputFolder  = 'jobConfig')
