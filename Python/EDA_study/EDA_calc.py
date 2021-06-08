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
from datetime import datetime
import multiprocessing

def calculate(func, args):
    result = func(*args)
    return result

def calculatestar(args):
    return calculate(*args)

def hand_norm(A):
    return m.sqrt(np.sum(A ** 2))

def hand_scalar_prod(A,B):
    prod = np.zeros((len(A)))
    k = 0
    for a,b in (zip(A,B)):
        prod[k]= a * b 
        k +=1
        
    return np.sum(prod)

def hand_dist(A,B, metric = 'euclidean'):
    dist = np.zeros((len(A),(len(A))))
    if metric == 'euclidean':
        for i in range(len(A)):
            for ii in range(len(B)):
                dist[ii,i] = m.sqrt(np.sum((A[i,:] - B[ii,:]) ** 2))

    if metric == 'cosine':
        for i in range(len(A)):
            for ii in range(len(B)):
                dist[ii,i] = 1 - (hand_scalar_prod(A[i,:],B[ii,:])/(hand_norm(A[i,:])*hand_norm(B[ii,:])))
            
    if metric == 'mahalanobis':
        concat = np.zeros((len(A)+len(B),len(A[0])))
        concat[:len(A)] = A
        concat[len(A):] = B        
        VI = np.linalg.inv(np.cov(concat.T)).T
        for i in range(len(A)):
            for ii in range(len(B)):
                dist[ii,i] = np.sqrt(np.dot(np.dot((A[i,:]-B[ii,:]),VI),(A[i,:]-B[ii,:]).T))
            
    return dist

def EDA_calc (data, metric='euclidean'):            
    dist = hand_dist(data, data, metric=metric)
    
    CP = 1/np.sum(dist,axis=0)

    GD = np.sum(CP)/(2*CP)

    return GD

def main(n_i,total, background_percent, test_size, dat_set_percent):
    
    current_time = datetime.now().strftime("%H:%M:%S")
    
    print('=> Iteration Number {}:         .Initiation time:'.format(n_i+1),current_time)

    # Divide data-set into training and testing sub-sets
    print('=> Iteration Number {}:         .Dividing training and testing sub-sets'.format(n_i+1))
    
    # Reducing background samples
    _,reduced_background = train_test_split(background, test_size=dat_set_percent)
    
    # Dividing training and test sub-sets
    background_seed, streaming_background = train_test_split(reduced_background, test_size=test_size)
    
    # Iserting the correct number of signal in training
    _,background_seed = train_test_split(background_seed, test_size=background_percent)    
    
    n_signal_samples = len(background_seed)*(1-background_percent)
    
    _,training_signal = train_test_split(signal, test_size=n_signal_samples/len(signal))
    
    # Iserting the correct number of signal in streaming
    
    _,streaming_background = train_test_split(streaming_background, test_size=background_percent)
    
    n_signal_samples = len(streaming_background)*(1-background_percent)
    
    _,streaming_signal = train_test_split(signal, test_size=n_signal_samples/len(signal))

    # Concatenating Signal and the Background sub-sets
    
    streaming_data = np.vstack((streaming_background,streaming_signal))
    training_data = np.vstack((background_seed,training_signal))    


    print("=> Iteration Number {}:             .Training shape: {}".format((n_i+1),training_data.shape))
    print("=> Iteration Number {}:             .Training Background shape: {}".format((n_i+1),background_seed.shape))
    print("=> Iteration Number {}:             .Training Signal shape: {}".format((n_i+1),training_signal.shape))
    print("=> Iteration Number {}:             .Streaming shape: {}".format((n_i+1),streaming_data.shape))
    print("=> Iteration Number {}:             .Streaming Background shape: {}".format((n_i+1),streaming_background.shape))
    print("=> Iteration Number {}:             .Streaming Signal shape: {}".format((n_i+1),streaming_signal.shape))

    # Normalize Data
    print('=> Iteration Number {}:         .Normalizing Data'.format(n_i+1))
    streaming_data = normalize(streaming_data,norm='max',axis=0)
    training_data = normalize(training_data,norm='max',axis=0)

    # Creating Labels
    print('=> Iteration Number {}:         .Creating Labels'.format(n_i+1))
    y_training =np.ones((len(training_data)))
    y_training[:len(background_seed)] = 0
    
    y_streaming =np.ones((len(streaming_data)))
    y_streaming[:len(streaming_background)] = 0

    # Training Naive Bayes
    print('=> Iteration Number {}:         .Training Naive Bayes'.format(n_i+1))
    clf = GaussianNB(priors=[0.99,0.01])
    clf.fit(training_data, y_training)

    prob = clf.predict_proba(streaming_data)[:,0]
    
    np.savetxt('bayes_prob_99_' + str(n_i) +'.csv', prob, delimiter=',')

    # Calculating EDA
    print('=> Iteration Number {}:         .Calculating EDA'.format(n_i+1))
    streaming_eda_bayes = prob * EDA_calc(streaming_data,metric ='euclidean')

    np.savetxt('streaming_eda_bayes_99_' + str(n_i) +'_euclidean.csv',streaming_eda_bayes,delimiter=',')

    streaming_eda_bayes = prob * EDA_calc(streaming_data,metric ='cosine')

    np.savetxt('streaming_eda_bayes_99_' + str(n_i) +'_cosine.csv',streaming_eda_bayes,delimiter=',')

    streaming_eda_bayes = prob * EDA_calc(streaming_data,metric ='mahalanobis')

    np.savetxt('streaming_eda_bayes_99_' + str(n_i) +'_mahalanobis.csv',streaming_eda_bayes,delimiter=',')

    ####################### Redivising Signal ##############################

    print('=> Iteration Number {}:         .Dividing training and testing sub-sets with 50%/50%')

    _,training_signal = train_test_split(signal, test_size=len(background_seed)/len(signal))

    # Concatenating Signal and the Background sub-sets
    
    training_data = np.vstack((background_seed,training_signal))    


    print("=> Iteration Number {}:             .Training shape 2: {}".format((n_i+1),training_data.shape))
    print("=> Iteration Number {}:             .Training Background shape 2: {}".format((n_i+1),background_seed.shape))
    print("=> Iteration Number {}:             .Training Signal shape 2: {}".format((n_i+1),training_signal.shape))

    # Normalize Data
    print('=> Iteration Number {}:         .Normalizing Data 2'.format(n_i+1))
    training_data = normalize(training_data,norm='max',axis=0)

    # Creating Labels
    print('=> Iteration Number {}:         .Creating Labels 2'.format(n_i+1))
    y_training =np.ones((len(training_data)))
    y_training[:len(background_seed)] = 0

    # Training Naive Bayes
    print('=> Iteration Number {}:         .Training Naive Bayes 2'.format(n_i+1))
    clf = GaussianNB(priors=[0.5,0.5])
    clf.fit(training_data, y_training)

    prob = clf.predict_proba(streaming_data)[:,0]
    
    np.savetxt('bayes_prob_50_' + str(n_i) +'.csv', prob, delimiter=',')

    # Calculating EDA
    print('=> Iteration Number {}:         .Calculating EDA 2'.format(n_i+1))
    streaming_eda_bayes = prob * EDA_calc(streaming_data,metric ='euclidean')

    np.savetxt('streaming_eda_bayes_50_' + str(n_i) +'_euclidean.csv',streaming_eda_bayes,delimiter=',')

    streaming_eda_bayes = prob * EDA_calc(streaming_data,metric ='cosine')

    np.savetxt('streaming_eda_bayes_50_' + str(n_i) +'_cosine.csv',streaming_eda_bayes,delimiter=',')

    streaming_eda_bayes = prob * EDA_calc(streaming_data,metric ='mahalanobis')

    np.savetxt('streaming_eda_bayes_50_' + str(n_i) +'_mahalanobis.csv',streaming_eda_bayes,delimiter=',')

##########################################################
# ------------------------------------------------------ #
# --------------------- INITIATION --------------------- #
# ------------------------------------------------------ #
##########################################################
### Define User Variables ###

PROCESSES = 4

# Number of Iterations
iterations = 33

# Number of events
total = 20000

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

with multiprocessing.Pool(PROCESSES) as pool:

        TASKS = [(main, (n_i,total, background_percent, test_size, dat_set_percent)) for n_i in range(iterations)]

        pool.map(calculatestar, TASKS)
