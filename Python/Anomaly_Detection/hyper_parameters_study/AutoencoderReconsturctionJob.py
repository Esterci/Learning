

__all__ = ['BinaryClassificationJob', 'lock_as_completed_job', 'lock_as_failed_job']

from autoencoder_Dijets import time_stamp
from Gaugi.messenger import Logger
from Gaugi.messenger.macros import *
from Gaugi import StatusCode, checkForUnusedVars, retrieve_kw
from six import print_

from tensorflow.keras.models import clone_model
from tensorflow.keras import backend as K
from sklearn.utils.class_weight import compute_class_weight

from datetime import datetime
from copy import deepcopy
import numpy as np
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import pandas as pd
from keras.optimizers import Adam
from Disco_tensor_flow import decorr
import matplotlib.pyplot as plt



def lock_as_completed_job(output):
  with open(output+'/.complete','w') as f:
    f.write('complete')


def lock_as_failed_job(output):
  with open(output+'/.failed','w') as f:
    f.write('failed')


def get_data( total, background_percent, test_size ):

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

    return train_data,test_data,test_df,test_labels


class AutoencoderReconsturctionJob( Logger ):

    def __init__(self , **kw ):

        Logger.__init__(self)

        self.epochs          = retrieve_kw( kw, 'epochs'         , 1000                  )
        self.batch_size      = retrieve_kw( kw, 'batch_size'     , 1024                  )
        self.lambda_disco    = retrieve_kw( kw, 'lambda_disco'     , 300                  )
        self.callbacks       = retrieve_kw( kw, 'callbacks'      , []                    )
        self.metrics         = retrieve_kw( kw, 'metrics'        , []                    )
        job_auto_config      = retrieve_kw( kw, 'job'            , None                  )        
        self.sorts           = retrieve_kw( kw, 'sorts'          , range(1)              )
        self.inits           = retrieve_kw( kw, 'inits'          , 1                     )
        self.__verbose       = retrieve_kw( kw, 'verbose'        , True                  )
        self.__model_generator=retrieve_kw( kw, 'model_generator', None                  )
        self.total = 100000
        self.background_percent = 0.99
        self.test_size = 0.3
    
        # read the job configuration from file
        if job_auto_config:
            if type(job_auto_config) is str:
                MSG_INFO( self, 'Reading job configuration from: %s', job_auto_config )
                from saphyra.core.readers import JobReader
                job = JobReader().load( job_auto_config )
            else:
                job = job_auto_config
        
            # retrive sort/init lists from file
            self.sorts = job.getSorts()
            self.inits = job.getInits()
            self.__models, self.__id_models = job.getModels()
            self.__jobId = job.id()

        # get model and tag from model file or lists
        models = retrieve_kw( kw, 'models', None )
        if models:
            self.__models = models
            self.__id_models = [id for id in range(len(models))]
            self.__jobId = 0


        checkForUnusedVars(kw)


    #
    # Sorts setter and getter
    #

    @property
    def sorts(self):
        return self.__sorts

    @sorts.setter
    def sorts( self, s):
        if type(s) is int:
            self.__sorts = range(s)
        else:
            self.__sorts = s


    #
    # Init setter and getter
    #
    @property
    def inits( self ):
        return self.__inits

    @inits.setter
    def inits( self, s):
        if type(s) is int:
            self.__inits = range(s)
        else:
            self.__inits = s

    def time_stamp(self):
        from datetime import datetime
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H.%M.%S")
        return timestampStr

    #
    # run job
    #
    def run( self ):



        for isort, sort in enumerate( self.sorts ):

            # get the current kfold and train, val sets
            train_data,test_data,test_df,test_labels = get_data(self.total, 
                                                                self.background_percent, 
                                                                self.test_size)
            for imodel, model in enumerate( self.__models ):

                for iinit, init in enumerate(self.inits):

                    struct_name = ( 'batch_size' + '__' + str(self.batch_size) + '__' +
                                    'lambda_disco' + '__' + str(self.lambda_disco) + '__' +
                                    'ID' + '__' + str(self.__jobId ) + '__' +
                                    time_stamp() + '__')

                    print(model)
                    # get the model "ptr" for this sort, init and model index
                    if self.__model_generator:
                        print(  "Apply model generator..." )
                        model_for_this_init = self.__model_generator( sort )
                    else: 
                        model_for_this_init = clone_model(model) # get only the model


                    try:
                        opt = Adam(lr=0.001)
                        model_for_this_init.compile( opt,
                                    loss=decorr(input_layer[:,15], 
                                                decoder[:,15], 
                                                sample_weights[:,0],
                                                self.lambda_disco),
                                    # protection for functions or classes with internal variables
                                    # this copy avoid the current training effect the next one.
                                    metrics = deepcopy(self.metrics),
                                    #metrics = self.metrics,
                                    )
                        model_for_this_init.summary()
                    except RuntimeError as e:
                        print("Compilation model error: %s" , e)


                    print("Training model id (%d) using sort (%d) and init (%d)", self.__id_models[imodel], sort, init )

                    callbacks = deepcopy(self.callbacks)

                    
                    # Hacn: used by orchestra to set this job as local test
                    if os.getenv('LOCAL_TEST'): 
                        print(  "The LOCAL_TEST environ was detected." )
                        print(  "This is a short local test, lets skip the fitting for now. ")
                        return StatusCode.SUCCESS

                    try:

                        Train = [train_data, np.ones(len(train_data))]

                        # Training
                        history = model_for_this_init.fit(x = Train, 
                                            y = train_data,
                                            epochs          = self.epochs,
                                            batch_size      = self.batch_size,
                                            verbose         = self.__verbose,
                                            validation_split= 0.1,
                                            callbacks       = callbacks,
                                            shuffle         = True).history


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

                        test_x_predictions = model_for_this_init.predict([test_data,
                                                                        np.ones(len(test_data))])
                        # Calculating MSE

                        end = datetime.now()

                        mse = np.mean(np.power(test_data - test_x_predictions, 2), 
                                    axis=1
                                    )
                
                        # Creating MSE data frame
                
                        error_df = pd.DataFrame({'reconstruction_error': mse,
                                                'class': test_labels,
                                                'time' : end -start
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
                        with open('Results/Error__' + struct_name + '.txt', 'w') as f:
                            f.write('error')


                    self.history = history

                    # Clear everything for the next init
                    K.clear_session()

            # Clear the keras once again just to be sure
            K.clear_session()

        return StatusCode.SUCCESS




