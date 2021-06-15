import numpy as np
import pandas as pd
import pickle
import math
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, cdist, squareform
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import os
import SODA
import data_manipulation as dm
import multiprocessing
from sklearn.utils.validation import check_array
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import math as m
from sklearn.naive_bayes import GaussianNB
from numba import njit

#-------------------------------------------------------------------------------------#
#-------------------------------------Main Code---------------------------------------#

def calculate(func, args):
    result = func(*args)
    return result

def calculatestar(args):
    return calculate(*args)

def kde_naive_bayes (X, kde_dict, prior):
    # calculate the independent conditional probability
    L,_ = np.shape(X)
    prob = np.ones((L)) * prior
        
    for i in range(L):
        for kde in kde_dict:
            prob[i] *= kde_dict[kde](X[i,kde])
    return prob

def kde_dictionary (X):
    _,W = np.shape(X)
    # Creating KDE dictionary
    s = set(range(W))
    kde_dict = dict.fromkeys(s)
    for kde in kde_dict:
        kde_dict[kde] = stats.gaussian_kde(X[:, kde])
    return kde_dict

def main():
    ##########################################################
    # ------------------------------------------------------ #
    # --------------------- INITIATION --------------------- #
    # ------------------------------------------------------ #
    ##########################################################
    
    ### Define Granularities to run
    granularity = [1,2,3,4]
    
    ### Define User Variables ###
    PROCESSES = 4

    # Number of Iterations
    iterations = 1

    # Number of events
    total = 100000

    # Percentage of background samples on the testing phase
    background_percent = 0.99

    # Percentage of samples on the training phase
    test_size = 0.3

    ##########################################################
    # ------------------------------------------------------ #
    # ----------------------- LOADING ---------------------- #
    # ------------------------------------------------------ #
    ##########################################################
    # Firstly the model loads the background and signal data, 
    # then it removes the attributes first string line, which 
    # are the column names, in order to avoid NaN values in 
    # the array.

    print('         ==== Commencing Initiation ====\n')

    ### Background
    b_name='Input_Background_1.csv'
    background = np.genfromtxt(b_name, delimiter=',')
    background = background[1:,:]
    Lb, W = background.shape
    print("     .Background Loaded..." )
    print("     .Background shape: {}".format(background.shape))

    ### Signal
    s_name='Input_Signal_1.csv'
    signal = np.genfromtxt(s_name, delimiter=',')
    signal = signal[1:,:]
    Ls, _ = signal.shape
    print("     .Signal Loaded...")
    print("     .Signal shape: {}\n".format(signal.shape))

    # Percentage of background samples to divide the data-set
    dat_set_percent = total/len(background)

    print('\n          ==== Initiation Complete ====\n')
    print('=*='*17 )
    print('      ==== Commencing Data Processing ====')


    for n_i in range(iterations):

        current_time = datetime.now().strftime("%H:%M:%S")
    
        print('=> Iteration Number {}:         .Initiation time:'.format(n_i+1),current_time)

        # Divide data-set into training and testing sub-sets
        print('=> Iteration Number {}:         .Dividing training and testing sub-sets'.format(n_i+1))
        
        # Reducing background samples
        _,reduced_background = train_test_split(background, test_size=dat_set_percent)
        
        # Dividing training and test sub-sets
        training_data, streaming_background = train_test_split(reduced_background, test_size=test_size)

        # Iserting the correct number of signal in streaming
        
        n_signal_samples = int(len(streaming_background)*(1-background_percent))

        _,streaming_background = train_test_split(streaming_background, test_size=background_percent)
        
        _,streaming_signal = train_test_split(signal, test_size=n_signal_samples/len(signal))

        # Concatenating Signal and the Background sub-sets
        
        streaming_data = np.vstack((streaming_background,streaming_signal))

        print("=> Iteration Number {}:             .Training shape: {}".format((n_i+1),training_data.shape))
        print("=> Iteration Number {}:             .Streaming shape: {}".format((n_i+1),streaming_data.shape))
        print("=> Iteration Number {}:             .Streaming Background shape: {}".format((n_i+1),streaming_background.shape))
        print("=> Iteration Number {}:             .Streaming Signal shape: {}".format((n_i+1),streaming_signal.shape))

        # Normalize Data
        print('=> Iteration Number {}:         .Normalizing Data'.format(n_i+1))
        streaming_data = normalize(streaming_data,norm='max',axis=0)
        training_data = normalize(training_data,norm='max',axis=0)

        # Creating Labels
        print('=> Iteration Number {}:         .Creating Labels'.format(n_i+1))
        
        y =np.ones((len(streaming_data)+len(training_data)))
        y[:len(training_data) + len(streaming_background)] = 0
        y[:len(training_data)] = -1
        np.savetxt('results/y_streaming.csv',y,delimiter=',')

        # Training Naive Bayes
        print('=> Iteration Number {}:         .Calculating Naive Bayes'.format(n_i+1))
        
        seed_kde_dict = kde_dictionary(training_data)

        prob = kde_naive_bayes(np.vstack((training_data,streaming_data)),seed_kde_dict,0.99)
        
        np.savetxt('results/posterior_probability__' + str(n_i) +'.csv', prob, delimiter=',')

        print('Creating pool with %d processes\n' % PROCESSES)

        with multiprocessing.Pool(PROCESSES) as pool:

            #
            # Tests

            TASKS = [(dm.SODA_Granularity_Iteration, (training_data,streaming_data, gra,prob,n_i)) for gra in granularity]

            pool.map(calculatestar, TASKS)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()       