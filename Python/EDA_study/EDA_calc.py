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
import multiprocessing
from datetime import datetime


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

def one_class_naive_bayes (X, kde_dict):
    
    # calculate the independent conditional probability
    L = len(X)
    prob = np.ones((L))
        
    for i in range(L):
        for kde in kde_dict:
            prob[i] *= kde_dict[kde](X[i])
    return prob

def kde_dictionary (X):
    _,W = np.shape(X)
    # Creating KDE dictionary
    s = set(range(W))
    kde_dict = dict.fromkeys(s)
    for kde in kde_dict:
        kde_dict[kde] = stats.gaussian_kde(X[:, kde])
    return kde_dict

def calculate(func, args):
    result = func(*args)
    return result

def calculatestar(args):
    return calculate(*args)

def main(n_i,total, background_percent, test_size, dat_set_percent):
    
    current_time = datetime.now().strftime("%H:%M:%S")
    
    print('=> Iteration Number {}:         .Initiation time:'.format(n_i+1),current_time)

    # Divide data-set into training and testing sub-sets

    print('=> Iteration Number {}:         .Dividing training and testing sub-sets'.format(n_i+1))
    
    # Reducing background samples
    _,reduced_background = train_test_split(background, test_size=dat_set_percent)
    
    # Dividing training and test sub-sets
    training_data, streaming_background = train_test_split(reduced_background, test_size=test_size)
    
    n_signal_samples = int(len(streaming_background)*(1-background_percent))

    # Iserting the correct number of signal in streaming

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
    
    # Concatenating the data-sets
    all_data = np.vstack((training_data,streaming_data))

    # Creating Labels
   
    print('=> Iteration Number {}:         .Creating Labels'.format(n_i+1))

    y = np.ones((len(all_data)))
    y[:len(training_data) + len(streaming_background)] = 0
    y[:len(training_data)] = -1

    np.savetxt('results_99/target.csv',y,delimiter=',')

    # Calculating EDA

    print('=> Iteration Number {}:         .Calculating EDA with euclidean'.format(n_i+1))
    eda = EDA_calc(all_data,metric ='euclidean')

    kde_dict = kde_dictionary(eda[:len(training_data)].reshape(-1,1))
    prob = one_class_naive_bayes(eda,kde_dict)

    np.savetxt('results_99/posterior_EDA__' + str(n_i) +'__euclidean.csv',prob,delimiter=',')

    posterior_eda = eda * prob

    np.savetxt('results_99/weighted_EDA__' + str(n_i) +'__euclidean.csv',posterior_eda,delimiter=',')


    # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

    print('=> Iteration Number {}:         .Calculating EDA with euclidean with cosine'.format(n_i+1))

    eda = EDA_calc(np.vstack((training_data,streaming_data)),metric ='cosine')

    kde_dict = kde_dictionary(eda[:len(training_data)].reshape(-1,1))
    prob = one_class_naive_bayes(eda,kde_dict)

    np.savetxt('results_99/posterior_EDA__' + str(n_i) +'__cosine.csv',prob,delimiter=',')

    posterior_eda = eda * prob

    np.savetxt('results_99/weighted_EDA__' + str(n_i) +'__cosine.csv',posterior_eda,delimiter=',')

    # =*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*

    print('=> Iteration Number {}:         .Calculating EDA with euclidean with mahalanobis'.format(n_i+1))
    
    eda = EDA_calc(np.vstack((training_data,streaming_data)),metric ='mahalanobis')

    kde_dict = kde_dictionary(eda[:len(training_data)].reshape(-1,1))
    prob = one_class_naive_bayes(eda,kde_dict)

    np.savetxt('results_99/posterior_EDA__' + str(n_i) +'__mahalanobis.csv',prob,delimiter=',')

    posterior_eda = eda * prob

    np.savetxt('results_99/weighted_EDA__' + str(n_i) +'__mahalanobis.csv',posterior_eda,delimiter=',')

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

print('\n          ==== Initiation Complete ====\n')
print('=*='*17 )
print('      ==== Commencing Data Processing ====')

##########################################################
# ------------------------------------------------------ #
# --------------------- INITIATION --------------------- #
# ------------------------------------------------------ #
##########################################################
### Define User Variables ###

# Number of Processes
PROCESSES = 4

# Number of Iterations
iterations = 33

# Number of events
total = 10000

# Percentage of background samples on the testing phase
background_percent = 0.99

# Percentage of samples on the training phase
test_size = 0.3

# Percentage of background samples to divide the data-set
dat_set_percent = total/len(background)

with multiprocessing.Pool(PROCESSES) as pool:

        TASKS = [(main, (n_i,total, background_percent, test_size, dat_set_percent)) for n_i in range(iterations)]

        pool.map(calculatestar, TASKS)