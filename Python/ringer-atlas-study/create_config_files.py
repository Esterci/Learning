import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv1D, Flatten

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

def get_model(min_batch_size,max_batch_size,min_hidden_dim,max_hidden_dim):
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

    modelCol = []

    for batch_size in batch_size_list:
        for encoding_dim in encoding_dim_list:
            for lambda_disco in lambda_disco_list:
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
                            
                            modelCol.append(autoencoder)
    
    return  #lista de NN keras


create_jobs(models = get_modelget_model(min_batch_size,
                                        max_batch_size,
                                        min_hidden_dim,
                                        max_hidden_dim), #passa lista de NN
            nInits = 1,
            nInitsPerJob = 1,
            sortBounds = 10,
            nSortsPerJob = 1,
            nModelsPerJob = 1,
            outputFolder = 'job_test2')
