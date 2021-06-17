import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pickle
import SODA
import threading
from datetime import datetime
from psutil import cpu_percent, swap_memory
from progress.bar import Bar
from sklearn.neighbors import kneighbors_graph 
from scipy.sparse.linalg import expm
import scipy.sparse 
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing
import sklearn
from sklearn.utils.validation import check_array

class performance(threading.Thread):
    # Declares variables for perfomance analysis:
    
    def __init__(self):
        threading.Thread.__init__(self)
        self.control = True
    
    def run(self):
        cpu_p = []
        ram_p = []
        ram_u = []
        while self.control:
            cpu_p.append(cpu_percent(interval=1, percpu=True))
            ram_p.append(swap_memory().percent)
            ram_u.append(swap_memory().used/(1024**3))
        self.mean_cpu_p = np.mean(cpu_p)
        self.mean_ram_p = np.mean(ram_p)
        self.mean_ram_u = np.mean(ram_u)
        self.max_cpu_p = np.max(np.mean(cpu_p, axis=1))
        self.max_ram_p = np.max(ram_p)
        self.max_ram_u = np.max(ram_u)
    
    def stop(self):
        self.control = False
    
    def join(self):
        threading.Thread.join(self)
        out = {'mean_cpu_p': self.mean_cpu_p,
               'mean_ram_p': self.mean_ram_p,
               'mean_ram_u': self.mean_ram_u,
               'max_cpu_p': self.max_cpu_p,
               'max_ram_p': self.max_ram_p,
               'max_ram_u': self.max_ram_u}
        return out

def divide(data, n_windows = 100, n_samples = 50):  
    """Divide the data in n_samples, and the samples are equaly distributed in n_windows
    -- Input
    - data = data to split
    - n_windows = number of windows to separete the data
    - n_samples = number of samples of the output
    -- Output
    - reduced_data = splited data with n_samples
    - data_sample_id = id of the splited data"""  
    L, W = data.shape
    
    # Checking if the windows can be of the same size 
    
    if int(L % n_windows) != 0:
        
        # Checking if we need to peak the same amount of data from each window
        
        if int(n_samples % n_windows) != 0 or (n_samples/n_windows) % 1 != 0:
            
            lines_per_window = L // n_windows
            samples_per_window = n_samples // n_windows
            reduced_data = np.zeros((int(n_samples),W))
            data_sample_id = np.zeros(int(n_samples))

            for i in range(n_windows):
                if i >= n_windows - 1:
                    for j in range(samples_per_window + int(n_windows*((n_samples/n_windows) % 1))):
                        sample = np.random.randint(i*lines_per_window,int((i+1)*lines_per_window + int(L % n_windows)))
                        new_line = data[sample]
                        reduced_data[j+(i*samples_per_window)] = new_line
                        data_sample_id[j+(i*samples_per_window)] = sample

                else:
                    for j in range(samples_per_window):
                        sample = np.random.randint(i*lines_per_window,(i+1)*lines_per_window)
                        new_line = data[sample]
                        reduced_data[j+(i*samples_per_window)] = new_line
                        data_sample_id[j+(i*samples_per_window)] = sample
            
        # Even amount of data 
            
        else:
            
            lines_per_window = L // n_windows
            samples_per_window = n_samples // n_windows
            reduced_data = np.zeros((int(n_samples),W))
            data_sample_id = np.zeros(int(n_samples))

            for i in range(n_windows):
                if i >= n_windows - 1:
                    for j in range(samples_per_window):
                        sample = np.random.randint(i*lines_per_window,int((i+1)*lines_per_window + int(L % n_windows)))
                        new_line = data[sample]
                        reduced_data[j+(i*samples_per_window)] = new_line
                        data_sample_id[j+(i*samples_per_window)] = sample

                else:
                    for j in range(samples_per_window):
                        sample = np.random.randint(i*lines_per_window,(i+1)*lines_per_window)
                        new_line = data[sample]
                        reduced_data[j+(i*samples_per_window)] = new_line
                        data_sample_id[j+(i*samples_per_window)] = sample 
                        
    # Windows of same size 

    else:
        
        # Checking if we need to pick the same amount of data from each window
        
        if int(n_samples % n_windows) != 0 or (n_samples/n_windows) % 1 != 0:
            
            lines_per_window = L // n_windows
            samples_per_window = n_samples // n_windows
            reduced_data = np.zeros((int(n_samples),W))
            data_sample_id = np.zeros(int(n_samples))
            
            for i in range(n_windows):
                if i >= n_windows - 1:
                    for j in range(samples_per_window + int(n_windows*((n_samples/n_windows) % 1))):
                        sample = np.random.randint(i*lines_per_window,int((i+1)*lines_per_window + int(L % n_windows)))
                        new_line = data[sample]
                        reduced_data[j+(i*samples_per_window)] = new_line
                        data_sample_id[j+(i*samples_per_window)] = sample

                else:
                    for j in range(samples_per_window):
                        sample = np.random.randint(i*lines_per_window,(i+1)*lines_per_window)
                        new_line = data[sample]
                        reduced_data[j+(i*samples_per_window)] = new_line
                        data_sample_id[j+(i*samples_per_window)] = sample
            
        # Even amount of data
            
        else:
        
            lines_per_window = L // n_windows
            samples_per_window = n_samples // n_windows
            reduced_data = np.zeros((int(n_samples),W))
            data_sample_id = np.zeros(int(n_samples))

            for i in range(n_windows):
                for j in range(samples_per_window):
                    sample = np.random.randint(i*lines_per_window,int((i+1)*lines_per_window))
                    new_line = data[sample]
                    reduced_data[j+(i*samples_per_window)] = new_line
                    data_sample_id[j+(i*samples_per_window)] = sample
        
    return reduced_data, data_sample_id

def Normalisation(data):
    """Use standart deviation to normalise the data
    -- Input
    - data
    -- Output
    - Normalised data
    """

    # Normalizing whole data
    scaler = StandardScaler().fit(data)
    norm_data = scaler.transform(data)

    return norm_data

def PCA_Analysis(mantained_variation, attributes_influence,laplace=True):
    """Create and save the PCA model
    -- Input
    - mantained_variation = variation mantained for each PC
    - attributes_influence = influence of each attribute on the model 
    -- Output
    - Saves plot figures in results folder
    """

    # Plots the variation mantained by each PC

    fig = plt.figure(figsize=[16,8])
    ax = fig.subplots(1,1)
    ax.bar(x=['PC' + str(x) for x in range(1,(len(mantained_variation)+1))],height=mantained_variation)

    ax.set_ylabel('Percentage of Variance Held',fontsize=20)
    ax.set_xlabel('Principal Components',fontsize=20)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=18)
    ax.grid()

    fig.savefig('results/Percentage_of_Variance_Held.png', bbox_inches='tight')
                        
    sorted_sensors_contribution = attributes_influence.values[:]      
                        
    # Ploting Cntribution Attributes influence
                        
    fig = plt.figure(figsize=[25,8])

    fig.suptitle('Attributes Weighted Contribution Percentage', fontsize=16)

    ax = fig.subplots(1,1)

    sorted_sensors_contribution = sorted_sensors_contribution.ravel()
    ax.bar(x=list(attributes_influence.columns),height=sorted_sensors_contribution)
    plt.ylabel('Relevance Percentage',fontsize = 20)
    plt.xlabel('Attributes',fontsize = 20)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=18)
    plt.xticks(rotation=90)
    ax.grid()
    
    fig.savefig('results/Attributes_Contribution.png', bbox_inches='tight')

    return

def PCA_Projection(background_train,streaming_data, N_PCs, maintained_features=0):
    """Transform Data with PCA and normalize
    -- Input
    - Offline data
    - Streaming data
    - N_PCs = number of PCs to calculate
    -- Output
    - Projected Offline data
    - Projected Streaming data
    - Variation Mantained
    - Attributes Influence
    """

    # Calcules the PCA and projects the data-set into them

    pca= PCA(n_components = N_PCs)
    pca.fit(background_train)
            
    # Calculates the total variance maintained by each PCs
            
    pca_variation = pca.explained_variance_ratio_ * 100
    

    print('Normal Variation maintained: %.2f' % np.round(pca_variation.sum(), decimals = 2))

    proj_background_train = pca.transform(background_train)
    proj_streaming_data = pca.transform(streaming_data)
 

    ### Attributes analyses ###

    columns=["px1","py1","pz1","E1","eta1","phi1","pt1",\
                "px2","py2","pz2","E2","eta2","phi2",\
                "pt2","Delta_R","M12","MET","S","C","HT",\
                "A", "Min","Max","Mean","Var","Skw","Kurt",\
                "M2","M3","M4"] #,"Bmin","Bmax"]

    # Gets eigen vectors information from the trained pca object
    eigen_matrix = np.array(pca.components_)

    # Inverting negative signals
    eigen_matrix = pow((pow(eigen_matrix,2)),0.5) 

    # Calculates the feature contribution

    for i in range (eigen_matrix.shape[0]):
        LineSum = sum(eigen_matrix[i,:])
        for j in range (eigen_matrix.shape[1]):
            eigen_matrix[i,j] = ((eigen_matrix[i,j]*100)/LineSum)

    weighted_contribution = np.zeros((eigen_matrix.shape[1]))

    for i in range (eigen_matrix.shape[1]):
        NumeratorSum = 0
        for j in range (N_PCs):
            NumeratorSum += eigen_matrix[j,i] * pca_variation[j]

        weighted_contribution[i] = NumeratorSum / sum(pca_variation)

    weighted_contribution = weighted_contribution.reshape((1,-1))

    # Sorting attributes by their contribution values 
                        
    attributes_contribution = pd.DataFrame (weighted_contribution, columns = columns)
                        
    attributes_contribution = attributes_contribution.sort_values(by=0, axis=1,ascending=False)

    return proj_background_train, proj_streaming_data, pca_variation, attributes_contribution

def statistics_attributes(data):
    """
    When xyz_attributes=True:
       Concatenate with the data, statistics attributes for each event.
       Currently Applied:
        - Minimum Value
        - Maximum Value
        - Mean
        - Variance
        - Skewness
        - Kurtosis
        - 2nd Central Moment
        - 3rd Central Moment
        - 4th Central Moment
        - Bayesian Confidence Interval (Min and Max)
        More attributes may be added later
       
       -- Input
       - data [numpy.array]
       -- Output
       - output_data = data with adition of statistical features for each line [numpy.array]
       """
    
    momentum_data = np.concatenate((data[:, 0:3], data[:, 7:10]), axis=1)

    L, W = data.shape

    _, (min_v, max_v), mean, var, skew, kurt = stats.describe(momentum_data.transpose())
    
    # Minimum Value
    min_v = min_v.reshape(-1,1)
    
    min_v = check_array(min_v) # Minimum Value

    # Maximum Value
    max_v = max_v.reshape(-1,1)

    max_v = check_array(max_v) # Maximum Value

    # Mean
    mean = mean.reshape(-1,1)
    
    mean = check_array(mean) # Mean

    # Variance
    var = var.reshape(-1,1) 
    
    var = check_array(var) # Variance

    # Skewness
    skew = skew.reshape(-1,1)
    
    skew = check_array(skew) # Skewness

    # Kurtosis
    kurt = kurt.reshape(-1,1)
    
    kurt = check_array(kurt) # Kurtosis

    # 2nd Central Moment
    moment2 = stats.moment(momentum_data.transpose(), moment=2).reshape(-1,1)
    
    moment2 = check_array(moment2) # 2nd Central Moment

    # 3rd Central Moment
    moment3 = stats.moment(momentum_data.transpose(), moment=3).reshape(-1,1)
    
    moment3 = check_array(moment3) # 3rd Central Moment

    # 4th Central Moment
    moment4 = stats.moment(momentum_data.transpose(), moment=4).reshape(-1,1)
    
    moment4 = check_array(moment4) # 4th Central Moment

    """
    bayes_min = np.zeros(L).reshape(-1,1)

    bayes_min = check_array(momentum_data) # bayes_min

    bayes_max = np.zeros(L).reshape(-1,1)

    bayes_max = check_array(momentum_data) # bayes_max

    for i,d in enumerate(momentum_data):
        bayes = stats.bayes_mvs(d)

        bayes = check_array(momentum_data) # bayes

        bayes_min[i] = bayes[0][1][0]

        bayes_min = check_array(momentum_data) # bayes_min

        bayes_max[i] = bayes[0][1][1]

        bayes_max = check_array(momentum_data) # bayes_max
    """   
    output_data = np.concatenate((data, min_v, max_v, mean, var, skew, kurt, moment2, moment3, moment4), axis=1)
                                        #bayes_min, bayes_max
    return output_data

def SODA_Granularity_Iteration(offline_data,streaming_data,gra,prob,Iteration):

    print('=> Iteration Number {}:         .Executing granularity:'.format(Iteration),gra)
    # Formmating  Data
    offline_data = np.matrix(offline_data)
    L1 = len(offline_data)

    streaming_data = np.matrix(streaming_data)
    
    data = np.concatenate((offline_data, streaming_data), axis=0)

    # Dreate data frames to save each iteration result.

    performance_info = pd.DataFrame(np.zeros((1,8)).reshape((1,-1)), columns=['Granularity', 'Time_Elapsed',
                                                                'Mean CPU_Percentage', 'Max CPU_Percentage',
                                                                'Mean RAM_Percentage', 'Max RAM_Percentage',
                                                                'Mean RAM_Usage_GB', 'Max RAM_Usage_GB'])

    begin = datetime.now()

    performance_thread = performance()
    performance_thread.start()
    
    Input = {'GridSize':gra, 'StaticData':np.vstack((offline_data,streaming_data)), 'DistanceType': 'euclidean','PosteriorProbability':prob}

    out = SODA.SelfOrganisedDirectionAwareDataPartitioning(Input,'Offline')

    performance_thread.stop()
    performance_out = performance_thread.join()
    final = datetime.now()
    performance_info.loc[0,'Time_Elapsed'] = (final - begin)
    performance_info.loc[0,'Mean CPU_Percentage'] = performance_out['mean_cpu_p']
    performance_info.loc[0,'Max CPU_Percentage'] = performance_out['max_cpu_p']
    performance_info.loc[0,'Mean RAM_Percentage'] = performance_out['mean_ram_p']
    performance_info.loc[0,'Max RAM_Percentage'] = performance_out['max_ram_p']
    performance_info.loc[0,'Mean RAM_Usage_GB'] = performance_out['mean_ram_u']
    performance_info.loc[0,'Max RAM_Usage_GB'] = performance_out['max_ram_u']

    np.savetxt('results/SODA_labels__'+ str(gra) + '__' + str(Iteration) +'__.csv', out['IDX'],delimiter=',')
    np.savetxt('results/Density_1__'+ str(gra) + '__' + str(Iteration) +'__.csv', out['Density1'],delimiter=',')
    np.savetxt('results/Density_2__'+ str(gra) + '__' + str(Iteration) +'__.csv', out['Density2'],delimiter=',')
    performance_info.to_csv('results/performance_info__' + str(gra) + '__' + str(Iteration) + '__.csv', index=False)
