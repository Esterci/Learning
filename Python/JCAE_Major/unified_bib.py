import numpy as np
import pandas as pd
import threading
import time
import pickle
import tsfresh
from psutil import cpu_percent
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from sklearn.decomposition import PCA
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy as sp
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler,normalize
from scipy import io
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, cdist, squareform
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import recall_score, f1_score, precision_score
from tsfresh.feature_extraction import extract_features, ComprehensiveFCParameters
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.feature_extraction import feature_calculators
from tsfresh.feature_extraction import EfficientFCParameters
import os
import glob
from tsfresh.feature_extraction import extract_features
import selection

def Recovery (DataName): #Recovery function 

    #Changing Work Folder

    add_path1 = "/PCA_Analyses/"
    add_path2 = "/Kernel/"
    add_path3 = "/.Recovery/"
    base_path = os.getcwd ()
    working_path = os.getcwd() + '/Model'
    PCA_Analyses_path = working_path + add_path1
    Kernel_path = working_path + add_path2
    Recovery_path = working_path + add_path3
    
    if DataName == 'D_S_parameters':

        try:

            # Now change to Kernel directory
    
            os.chdir( Kernel_path )
            
            Final_Target = np.genfromtxt('FinalTarget.csv', delimiter = ',')
            
            # Now change to Recovery directory
        
            os.chdir( Recovery_path )
            
            P_N_groups = int(np.load('M_N_groups.npy'))
            Output_Id = int(np.load('ID.npy'))
            P_N_Ids = int(np.load('N_IDs.npy'))
            
            # Now change to base directory
        
            os.chdir( base_path )
            
            Output = {'FinalTarget': Final_Target,
                    'M_N_groups': P_N_groups,
                    'ID': Output_Id,
                    'N_IDs': P_N_Ids}
            
            #retval = os.getcwd()
            #print ("Final working directory %s" % retval)
            print("D_S_parameters Recovered!")
        
            return Output
            
        except:

            print("D_S_parameters not recovered =(" + '\033[0m')
    
    elif DataName == 'ExtractedNames':
        
        try:

            # Now change to Recovery directory
        
            os.chdir( Recovery_path )
            
            extracted_names = np.load('extracted_names.npy')
            
            # Now change to base directory
        
            os.chdir( base_path )
            
            #retval = os.getcwd()
            #print ("Final working directory %s" % retval)

            print("ExtractedNames recovered!")
            return extracted_names

        except:
            print('\033[93m' + "ExtractedNames not recovered =(" + '\033[0m')
                 
    elif DataName == 'SelectedFeatures':

        try:    
            # Now change to Recovery directory
        
            os.chdir( Recovery_path )
            
            Output_Id = int(np.load('ID.npy'))
            
            # Now change to Kernel directory
        
            os.chdir( Kernel_path )
            
            features_filtered_1 = pd.read_csv('features_filtered_' + str(Output_Id) + '.csv') 
            
            # Now change to base directory
        
            os.chdir( base_path )
            
            Output = {'FeaturesFiltered': features_filtered_1,
                    'ID': Output_Id}
            
            #retval = os.getcwd()
            #print ("Final working directory %s" % retval)

            print("SelectedFeatures recovered!")            
            return Output

        except:
            print('\033[93m' + "SelectedFeatures not recovered =(" + '\033[0m')
        
    elif DataName == 'ReducedFeatures':
        
        try:

            # Now change to Recovery directory
        
            os.chdir( Recovery_path )
            
            Output_Id = int(np.load('ID.npy'))
            
            # Now change to PCA Analyses directory
        
            os.chdir( PCA_Analyses_path )
            
            features_reduzidas = np.genfromtxt("features_reduzidas_" + str(Output_Id) + ".csv", delimiter=',')
            
            # Now change to base directory
        
            os.chdir( base_path )
            
            Output = {'ReducedFeatures': features_reduzidas,
                    'ID': Output_Id}
            
            #retval = os.getcwd()
            #print ("Final working directory %s" % retval)
            
            print("ReducedFeatures recovered!")
            return Output

        except:
            print('\033[93m' + "ReducedFeatures not recovered =(" + '\033[0m')
        
    elif DataName == 'SODA_parameters_processing_parameters':
        
        try:

            # Now change to base directory
        
            os.chdir( Recovery_path )

            #retval = os.getcwd()
            #print ("Current working directory %s" % retval)
            
            Output_Id = int(np.load('ID.npy'))
            processing_parameters = np.load(('processing_parameters.npy'), allow_pickle=True) 
            processing_parameters = processing_parameters.tolist() 
            distances = np.load(('distances.npy'), allow_pickle=True) 
            distances = distances.tolist() 
            min_granularity = np.load('Min_g.npy') 
            max_granularity = np.load('Max_g.npy') 
            pace = np.load('Pace.npy') 

            Output = {'Distances': distances,
                    'Min_g': min_granularity,
                    'Max_g': max_granularity,
                    'Pace': pace,
                    'ID': Output_Id}

            # Now change to base directory

            os.chdir( base_path ) 

            #retval = os.getcwd()
            #print ("Current working directory %s" % retval)

            print("SODA_parameters_processing_parameters recovered!")
            return Output, processing_parameters

        except:
            print('\033[93m' + "SODA_parameters_processing_parameters not recovered =(" + '\033[0m')
                
    elif DataName == 'ClassificationPar':

        try:
        
            # Now change to base directory
        
            os.chdir( Recovery_path ) 

            #retval = os.getcwd()
            #print ("Current working directory %s" % retval)
            
            Output_Id = int(np.load('ID.npy'))
            pace = np.load("Pace.npy")
            distances = np.load(('distances.npy'), allow_pickle=True) 
            distances = distances.tolist() 
            define_percent = np.load('define_percent.npy')
            
            Output = {'Percent': define_percent,
                    'Distances': distances,
                    'Pace': pace,
                    'ID': Output_Id}
            
            # Now change to base directory

            os.chdir( base_path ) 

            #retval = os.getcwd()
            #print ("Current working directory %s" % retval)
            
            print("ClassificationPar recovered!")
            return Output

        except:
            print('\033[93m' + "ClassificationPar not recovered =(" + '\033[0m')    

    elif DataName == 'ModelPar':
        
        try:

            # Now change to base directory
        
            os.chdir( Recovery_path ) 

            #retval = os.getcwd()
            #print ("Current working directory %s" % retval)
            
            # load the model from disk
            model = pickle.load(open("Model.sav", 'rb'))
            X_test = np.load('X_test.npy') 
            y_test = np.load('y_test.npy') 

            Output = {'Model': model,
                    'X': X_test,
                    'Y': y_test}
            
            # Now change to base directory

            os.chdir( base_path ) 

            #retval = os.getcwd()
            #print ("Current working directory %s" % retval)
            
            print("ModelPar recovered!")
            return Output

        except:
            print('\033[93m' + "ModelPar not recovered =(" + '\033[0m')   
        
    else:
        print('\033[93m' + "Wrong name lad/lass, please check de Recovery input" + '\033[0m')
    
def scale(X, x_min, x_max): #Normalization

    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    if denom==0:
        denom = 1
    return x_min + nom/denom

def format_func(value, tick_number): #Plot Formater
    # find number of multiples of pi/2
    N = int(value)
    if N == 0:
        return "X1"
    elif N == 50:
        return "X50"
    elif N == 100:
        return "X100"
    elif N == 150:
        return "X150"
    elif N == 200:
        return "X200"
    elif N == 250:
        return "X250"
    elif N == 300:
        return "X300"
    elif N == 350:
        return "X350"
    elif N == 400:
        return "X400"
    elif N == 450:
        return "X450"
    elif N == 500:
        return "X500"
    elif N == 550:
        return "X550"
    elif N == 600:
        return "X600"
    elif N == 650:
        return "X650"
    elif N == 700:
        return "X700"
    elif N == 750:
        return "X750"
    elif N == 800:
        return "X800"
    elif N == 850:
        return "X850"

def DataSlicer (Output_Id, id_per_group=20, Choice='All'):
    ''' Function to Slice a time series dataset into several datasets
    for save RAM during model execution
    
    Parameters:
    ------
    Output_Id : int
        identifier for the dataset
    
    id_per_group: int, optional
        number of time series per division (default is 20)
    
    Choice : str, optional
        option of data, can be ['Main Data', 'Eminence Data', 'All'] (default is 'All')
    
    
    Returns: 
    -------
    dictionary, with the following items
        'FinalTarget': np.array
            targets of the entire dataset
        'M_N_groups': int
            number of groups
        'ID': int
            identifier for the dataset
        'N_IDs': int
            number of time series
    
    '''
    
    print('Data Slicer Control Output')
    print('----------------------------------')
    
    #Changing Work Folder
    
    add_path1 = "/Input/"
    add_path2 = "/Kernel/"
    add_path3 = "/.Recovery/"
    base_path = os.getcwd()
    working_path = os.getcwd()
    Input_path = working_path + add_path1
    Kernel_path = working_path + add_path2
    Recovery_path = working_path + add_path3
     
    # Now change to Input directory
    
    os.chdir( Input_path )
    
    # Loading the required input 
    
    Full_data = np.genfromtxt('Output_' + str(int(Output_Id)) + '.csv', delimiter=',')
    #E_data = np.genfromtxt('Eminence_Data_' + str(Output_Id) + '.csv', delimiter=',')
    columns = Full_data.shape[1]
    data = Full_data[:,2:columns-1]
    info = Full_data[:,0:2]
    #centralizar os dados e colocá-los com desvioPadrão=1
    #scaler = MinMaxScaler(feature_range=(-1,1)).fit(data)
    #data = scaler.transform(data)
    
    
    P_data = np.concatenate((info,data), axis=1)
    
    Target = Full_data[:,columns-1]

    print('Full Matrix: ' + str(Full_data.shape))
    print('Main Data: ' + str(P_data.shape))
    print('Labels: ' + str(Target.shape))
    #print('Eminence Data: ' + str(E_data.shape))
    
    # Now change to Kernel directory
          
    os.chdir( Kernel_path )
    
    #pickle.dump(scaler, open('norm.sav', 'wb'))

    ###______________________________________________________________________###
    ###                     ProDiMes Slicing Parameters                      ###


    P_N_Ids = int(np.amax(P_data,axis=0)[0])
    P_N_voos = int(np.amax(P_data,axis=0)[1])
    P_last_group = int(P_N_Ids % id_per_group)

    if P_last_group != 0:
        P_N_groups = int((P_N_Ids / id_per_group) + 1)
    else:
        P_N_groups = int (P_N_Ids / id_per_group)

    ### Formating Final Target ###

    Final_Target = np.zeros((P_N_Ids))
    p_n_good = 0
    p_n_bad = 0
    aquired_time = P_N_Ids*P_N_voos/1000
    for i in range (P_N_Ids):
        if Target [i*P_N_voos] == 0:
            p_n_good += 1
        else:
            p_n_bad += 1

        Final_Target[i] = Target [i*P_N_voos]

    print ('Total Number of Ids: ' + str(P_N_Ids))
    print ('Number of healthy Ids: ' + str(p_n_good))
    print ('Number of falty Ids: ' + str(p_n_bad))
    print ('Total lifetime: ' + str(aquired_time) + ' s')
    print ('Main data Number of mesures: ' + str(P_N_voos ))
    print ('Main data Number of groups: ' + str(P_N_groups ))
    print ('Main data Last group: ' + str(P_last_group ))
    print ('___________________________________________')

    ###______________________________________________________________________###
    ###                    Eminences Slicing Parameters                      ###

    #E_N_Ids = int(np.amax(E_data,axis=0)[0] - np.amax(P_data,axis=0)[0])
    #E_N_voos = int(np.amax(E_data,axis=0)[1]) + 1
    #E_last_group = int(E_N_Ids % id_per_group)

    #if (E_last_group != 0):
    #    E_N_groups = int((E_N_Ids / id_per_group) + 1)
    #else:
    #    E_N_groups = int (E_N_Ids / id_per_group)

    #print ('Eminences Number of Ids: ' + str(E_N_Ids ))
    #print ('Eminences Number of flights: ' + str(E_N_voos ))
    #print ('Eminences Number of groups: ' + str(E_N_groups ))
    #print ('Eminences Last group: ' + str(E_last_group ))

        
    #np.savetxt(('Target_' + str(int(Output_Id)) + '.csv'), Final_Target, delimiter = ',')
    
    
    ###______________________________________________________________________###
    ###                      Slicing Prodimes Data                           ###

    if (Choice =='Main Data') or (Choice =='All'):
    
        for i in range (P_N_groups):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        
            Data = np.zeros(((id_per_group * P_N_voos),columns-1))
        
            for j in range (id_per_group):
            
                for k in range (P_N_voos):
            
                    if (i  < (P_N_groups - 1)):
                        Data[(j * P_N_voos) + k,:] = P_data [(((i * id_per_group + j) * P_N_voos) + k ) ,:]

                    elif (P_last_group == 0) and (i == (P_N_groups - 1)):
                        Data[(j * P_N_voos) + k,:] = P_data [(((i * id_per_group + j) * P_N_voos) + k ) ,:]
            
            if (P_last_group != 0) and (i == (P_N_groups - 1)):     

                Data = np.zeros(((P_last_group * P_N_voos),columns-1))
            
                for j in range (P_last_group):
    
                    for k in range (P_N_voos):
    
                        Data[(j * P_N_voos) + k,:] = P_data [(((i * id_per_group + j) * P_N_voos) + k ) ,:]
        
            np.savetxt(('Data_' + str(i) + '.csv'), Data, delimiter = ',')
    
    ###______________________________________________________________________###
    ###                          Slicing Eminences                           ###
    '''
    if (Choice == 'Eminence Data') or (Choice =='All'):
    
        for i in range (E_N_groups):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        
            Data = np.zeros(((id_per_group * E_N_voos),columns-3))
        
            for j in range (id_per_group):
            
                for k in range (E_N_voos):
            
                    if (i  < (E_N_groups - 1)):
                        Data[(j * E_N_voos) + k,:] = E_data [(((i * id_per_group + j) * E_N_voos) + k ) ,:]
                
        
            if (E_last_group != 0) and (i == (E_N_groups - 1)):
            
                Data = np.zeros(((E_last_group * E_N_voos),columns-3))
            
                for j in range (E_last_group):
    
                    for k in range (E_N_voos):
    
                        Data[(j * E_N_voos) + k,:] = E_data [(((i * id_per_group + j) * E_N_voos) + k ) ,:]
    
    
            np.savetxt(('Eminence_' + str(i) + '.csv'), Data, delimiter = ',')
    '''

    np.savetxt(('FinalTarget.csv'), Final_Target, delimiter = ',')
    
    # Now change to Recovery directory
          
    os.chdir( Recovery_path )
    
    np.save(('M_N_groups.npy'), P_N_groups)
    np.save(('ID.npy'), Output_Id)
    np.save(('N_IDs.npy'), P_N_Ids)
    
    # Now change back to Base directory
          
    os.chdir( base_path )
    
    Output = {'FinalTarget': Final_Target,
              'M_N_groups': P_N_groups,
              'ID': Output_Id,
              'N_IDs': P_N_Ids}
    
    return Output

def TSFRESH_Extraction(D_S_parameters):
    ''' Function to extract features of the time series using
    TSFRESH method
    
    Parameters:
    ------
    D_S_parameters : dictionary, with the following items
        'FinalTarget': np.array
            targets of the entire dataset
        'M_N_groups': int
            number of groups
        'ID': int
            identifier for the dataset
        'N_IDs': int
            number of time series
    
    
    Returns: 
    -------
    list
        a list of string with the name of the extracted features by TSFRESH
        
    '''
    
    print('             ')
    print('TSFRESH Control Output')
    print('----------------------------------')
    
    #Changing Work Folder
    
    add_path2 = "/Kernel/"
    add_path3 = "/.Recovery/"
    base_path = os.getcwd()
    working_path = os.getcwd()
    Kernel_path = working_path + add_path2
    Recovery_path = working_path + add_path3
        
    # Now change to Kernel directory
    
    os.chdir( Kernel_path )
    
    ###______________________________________________________________________###
    ###                         Feature Extraction                           ###

    #E_N_groups = np.load('E_N_groups.npy')
    P_N_groups = D_S_parameters['M_N_groups']
    
    for i in range(P_N_groups):
        
        Data = np.genfromtxt('Data_' + str(i) + '.csv', delimiter=',')
        data = pd.DataFrame(Data, columns= ['id','time'] + ['Sensor_' + str(x) for x in range(1,(Data.shape[1]-1))])
        
        Data_extracted_features = extract_features(data,column_id = "id", column_sort="time",n_jobs=4,disable_progressbar=True)
        extracted_names = list(Data_extracted_features.columns)
        np.savetxt('Data_Features_' + str(i) + '.csv', Data_extracted_features.values, delimiter=',')
        
    #for i in range(E_N_groups):

    
    #    data = pd.DataFrame(np.genfromtxt('Eminence_' + str(i) + '.csv', delimiter=','), 
    #                        columns= ['id','time','sensor_1','sensor_2','sensor_3','sensor_4',
    #                                            'sensor_5','sensor_6','sensor_7'])
    #    extracted_features = extract_features(data, column_id = "id", column_sort="time")
    #    np.savetxt('Eminence_Features_' + str(i) + '.csv', extracted_features, delimiter=',')
    
    # Now change to Recovery directory
    
    os.chdir( Recovery_path )
    
    np.save('extracted_names.npy',extracted_names)
    
    # Now change back to base directory
    
    os.chdir( base_path )
    
    print("Number of Extracted Features: {}".format(len(extracted_names)))
    
    return extracted_names

def tsfresh_chucksize(full_data,output_id):
    # Loading the required input 
    
    L, W = full_data.shape

    data = full_data[:,2:-1]
    info = full_data[:,0:2]

    # Normalizing
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    n_measures = int(max(info[:,1]))

    target = full_data[::n_measures,-1]

    u, idx = np.unique(info[:,0], return_index=True)

    df = pd.DataFrame(np.concatenate((info,data), axis=1), columns= ['id','time'] + 
                        ['Sensor_' + str(x) for x in range(1,W-2)])
    
    with open('Kernel/valid_features_dict.pkl', 'rb') as f:
        kind_to_fc_parameters = pickle.load(f)
    
    columns = []
    
    for i,x in enumerate(kind_to_fc_parameters):
        aux = pd.DataFrame(np.hstack((df.loc[:,:'time'].values,
                            df.loc[:,x].values.reshape((-1,1)))),
                            columns=['id','time',x])
        
        aux2 = tsfresh.extract_features(aux, column_id="id", column_sort="time",
                                        default_fc_parameters=kind_to_fc_parameters[x],
                                        #chunksize=3*24000, 
                                        n_jobs=4,
                                        disable_progressbar=False)
        for j in range(len(aux2.columns.tolist())):columns.append(aux2.columns.tolist()[j])

        if i == 0:
            extracted_features = np.array(aux2.values)
        else:
            extracted_features = np.hstack((extracted_features,aux2.values))

    final_features = pd.DataFrame(extracted_features,columns=columns)

    filtered_features, relevance_table = selection.select_features(final_features, target, n_jobs=4)
    
    filtered_features.sort_index(inplace = True)
    
    with open('Kernel/final_target_' + output_id + '.pkl', 'wb') as f:
        pickle.dump(target, f)

    # Extracting the selected features dictionary from pandas data frame

    kind_to_fc_parameters = tsfresh.feature_extraction.settings.from_columns(filtered_features)

    # Saving dictionary for the on-line phase
    
    with open('Kernel/kind_to_fc_parameters.pkl', 'wb') as f:
        pickle.dump(kind_to_fc_parameters, f)
    
    with open('Kernel/columns.pkl', 'wb') as f:
        pickle.dump(filtered_features.columns.to_list(), f)
        
    Output = {'FeaturesFiltered': filtered_features,
              'FinalTarget': target,
              'RelevanceTable': relevance_table,
              'ID': int(output_id)}
    
    return Output

def tsfresh_chucksize_test(output_id):
    # Loading the required input 
    
    full_data = np.genfromtxt('Input/Output_' + output_id + '.csv',
                                delimiter=',')
    
    L, W = full_data.shape

    data = full_data[:,2:-1]
    info = full_data[:,0:2]
    
    # Normalizing
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    n_measures = int(max(info[:,1]))

    target = full_data[::n_measures,-1]

    u, idx = np.unique(info[:,0], return_index=True)

    df = pd.DataFrame(np.concatenate((info,data), axis=1), columns= ['id','time'] + 
                        ['Sensor_' + str(x) for x in range(1,W-2)])
    
    extracted_features = tsfresh.extract_features(df, column_id="id", column_sort="time", n_jobs=4)
    
    return extracted_features

def tsfresh_NaN_filter(output_id,fft=False):
    """
    Given an output_id, this function
    withdraw all NaN features from the 
    TSFRESH extraction; 

    Inputs: 
        -output_id: str() -> the given id
        -fft: True or False -> filter fft features
    
    Outputs:
        - Saves via picklen in ./Kernel/ 
        an extraction dictonary without 
        features that generates NaN
    """

    df = tsfresh_chucksize_test(output_id)
    features = df.columns
    nan_columns = []
    for col in features:
        data = df.loc[:,col].values
        nan_test = np.isnan(data)
        aux  = col.split('__')[1].split('_')[0]
        if aux == 'fft' and fft == True:
            nan_columns.append(col)
        
        elif any(nan == True for nan in nan_test):
            nan_columns.append(col)

    print('Percentage of invalid features: ', len(nan_columns)*100/len(features))

    valid_features = []

    for i in range(len(features)):
        if features[i] not in nan_columns:
            valid_features.append(features[i])
            
    print('Percentage of valid features: ', len(valid_features)*100/len(features))

    valid_features_dict = from_columns(valid_features)

    with open('Kernel/valid_features_dict.pkl', 'wb') as f:
            pickle.dump(valid_features_dict, f)

    return

def tsfresh_ensemble(output_id):
    # Loading the required input 
    full_data = np.genfromtxt('Input/Output_{}.csv'.format(output_id),
                                delimiter=',')
    
    L, W = full_data.shape
    
    data = full_data[:,2:-1]
    info = full_data[:,0:2]
    
    n_measures = int(max(info[:,1]))
    n_timeseries = int(max(info[:,0]))
    
    label = full_data[::n_measures,-1]
    
    scaler = MinMaxScaler(feature_range=(-1,1)).fit(data)
    data = scaler.transform(data)
    
    with open('Kernel/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    full_data = np.concatenate((info,data), axis=1)
    
    divisions = 1
    idx = np.random.choice(range(n_timeseries),n_timeseries,replace=False)
    idx_division = np.array_split(idx,divisions)
    
    for i,div in enumerate(idx_division):
        div.sort()
        indices = [d2 for d1 in div for d2 in range(d1*n_measures,(d1+1)*n_measures)]
        ensemble_data = full_data[indices,:]
        ensemble_label = label[div]
        
        df = pd.DataFrame(ensemble_data, columns= ['id','time'] + 
                            ['Sensor_' + str(x) for x in range(1,W-2)])
    
        extracted_features = tsfresh.extract_features(df, column_id="id", column_sort="time", n_jobs=0)
        
        features = extracted_features.columns

        nan_columns = []
        for col in features:
            nan_test = np.isnan(extracted_features.loc[:,col].values)
            if any(nan == True for nan in nan_test):
                nan_columns.append(col)

        print(' - Percentage of invalid features: ', len(nan_columns)*100/len(features))

        cleaned_features = features.drop(nan_columns)
        cleaned_df = extracted_features[cleaned_features]

        filtered_df, relevance_table = selection.select_features(cleaned_df, ensemble_label, n_jobs=0)

        relevance_table.fillna(value=100)
        if i == 0:
            relevance_table_final = relevance_table.copy()
            extracted_features_final = extracted_features.copy()
            
        else:
            relevance_table_final.p_value = relevance_table_final.p_value + relevance_table.p_value
            extracted_features_final = pd.concat([extracted_features_final,extracted_features], axis=0)

        extracted_features_final = extracted_features_final.sort_index()
        
        
    relevance_table_final.p_value = relevance_table_final.p_value/divisions
    relevance_table_final.relevant = relevance_table_final.p_value < 0.0029
      
        
    relevant_features = relevance_table_final[relevance_table_final.relevant].feature
    extracted_features_final = extracted_features_final[relevant_features]
    
    kind_to_fc_parameters = from_columns(relevant_features)
    
    with open('Kernel/kind_to_fc_parameters.pkl', 'wb') as f:
        pickle.dump(kind_to_fc_parameters, f)
    
    with open('Kernel/columns.pkl', 'wb') as f:
        pickle.dump(relevant_features.keys().tolist(), f)

    with open('Kernel/final_target_{}.pkl'.format(output_id), 'wb') as f:
        pickle.dump(label, f)

    Output = {'FeaturesFiltered': extracted_features_final,
              'FinalTarget': label,
              'ID': int(output_id)}
    
    return Output

def dynamic_tsfresh (total_data, mode='prototype'):
    ''' Function for ONLINE mode
    This function read the data from the acquisition module and executes a 
    dynamic and lighter version of TSFRESH.
    
    Parameters:
    ------
    output_id : int 
        identifier of the seed dataset
    
    extracted_names: list
    
    Returns: 
    -------
    dataframe #########################################################
        

    '''
        

    data = total_data[:,2:-1]
    info = total_data[:,0:2]
        
    # Normalizing
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    total_data = np.concatenate((info,data), axis=1)
      
    # ----------------------------------------------------------------- # 
    df = pd.DataFrame(total_data, columns= ['id','time'] + 
                        ['Sensor_' + str(x) for x in range(1,(total_data.shape[1]-1))])
    
    # Loading feature dictionary
    with open('Kernel/kind_to_fc_parameters.pkl', 'rb') as f:
        kind_to_fc_parameters = pickle.load(f)
    
    # Loading column names

    with open('Kernel/columns.pkl', 'rb') as f:
        original_columns = pickle.load(f)
    
    columns = []
    

    for i,x in enumerate(kind_to_fc_parameters):
        aux = pd.DataFrame(np.hstack((df.loc[:,:'time'].values,
                            df.loc[:,x].values.reshape((-1,1)))),
                            columns=['id','time',x])
        
        aux2 = tsfresh.extract_features(aux, column_id="id", column_sort="time",
                                        default_fc_parameters=kind_to_fc_parameters[x],#chunksize=24000, 
                                        n_jobs=0
                                        #disable_progressbar=True
                                        )
        for j in range(len(aux2.columns.tolist())):columns.append(aux2.columns.tolist()[j])

        if i == 0:
            extracted_features = np.array(aux2.values)
        else:
            extracted_features = np.hstack((extracted_features,aux2.values))

    final_features = pd.DataFrame(extracted_features,columns=columns)
    final_features = final_features[original_columns]

    return impute(final_features), extracted_features

def test_tsfresh (SelectedFeatures,extracted_features):
    tsf_offline = SelectedFeatures['FeaturesFiltered'].values 
    tsf_online = extracted_features.values
    equal = np.equal(tsf_offline,tsf_online)
    n_errors = 0
    error_size = []
    for i in range(equal.shape[0]):
        for j in range(equal.shape[1]): 
            if equal[i,j]== False:
                n_errors += 1
                error_size.append(100*(tsf_offline[i,j]-tsf_online[i,j])/tsf_online[i,j])
    error_size = pd.DataFrame(error_size)            
    error_size = impute(error_size)

    print('Porcentagem de amostrar erradas (%): ',n_errors*100/(equal.shape[0]*equal.shape[1]))
    print('Média de erro percentual (%): ',np.mean(error_size[0]))
    print('Desvio (%): ',np.std(error_size[0]))

def PCA_calc (SelectedFeatures,N_PCs,Chose = 'Analytics',it=0):
    ''' Function to project and execute a Principal Components Analysis
    
    Parameters:
    ------
    SelectedFeatures : dictionary, with the following items
        'FeaturesFiltered': pd.DataFrame
            contain the output data of TSFRESH, i.e., the dataset with features selected by the hypothesis test
        'FinalTarget': np.array
            targets of the entire dataset
        'ID': int
            identifier for the dataset
    
    N_PCs: int
        number of Principal Components to mantain
    
    Chose: str
        type of analysis, can be ['Test', 'Calc', 'Specific', 'Analytics'] 
        (default is 'Analytics')
    
    Returns: 
    -------
    dictionary, with the following items
        'ReducedFeatures': np.array
            contain the output data of PCA, i.e., the dataset with Principal Componentes projected by PCA
        'ID': int
            identifier for the dataset
    
    '''

    if (Chose == 'Test') or (Chose == 'Calc') or (Chose == 'Specific') or (Chose == 'Analytics'):
        
        #Changing Work Folder
        
        add_path1 = "/PCA_Analyses/"
        add_path2 = "/Input/"
        add_path3 = "/Kernel/"
        add_path4 = "/PCA_Analyses/Figures/"        
        base_path = os.getcwd()
        working_path = os.getcwd()
        PCA_Analyses_path = working_path + add_path1
        Input_path = working_path + add_path2
        Kernel_path = working_path + add_path3
        PCA_Figures_path = working_path + add_path4
        
        # Now change to PCA Figures directory

        os.chdir( Kernel_path )
        
        print('             ')
        print('PCA Control Output')
        print('----------------------------------')

        Output_Id = SelectedFeatures['ID']
        features = SelectedFeatures['FeaturesFiltered']
        Target = SelectedFeatures['FinalTarget']
        selected_names = list(features.columns)

        #centralizar os dados e colocá-los com desvioPadrão=1
        scaler = StandardScaler().fit(features)
        features_padronizadas = scaler.transform(features)
        #features_padronizadas = pd.DataFrame(features_padronizadas)
        pickle.dump(scaler, open('pca_scaler.sav', 'wb'))

        pca= PCA(n_components = N_PCs)
        pca.fit(features_padronizadas)
        
        # save the model to disk

        pickle.dump(pca, open('pca.sav', 'wb'))
        
        variacao_percentual_pca = np.round(pca.explained_variance_ratio_ * 100, decimals = 2)
        
        # Now change to PCA Figures directory
        
        fig = plt.figure(figsize=[16,8])
        ax = fig.subplots(1,1)
        ax.bar(x=['PC' + str(x) for x in range(1,(N_PCs+1))],height=variacao_percentual_pca[0:N_PCs])

        ax.set_ylabel('Percentage of Variance Held',fontsize=27)
        ax.set_xlabel('Principal Components',fontsize=20)
        ax.tick_params(axis='x', labelsize=22)
        ax.tick_params(axis='y', labelsize=22)
        ax.grid()
        #plt.show()
        fig.savefig('Percentage_of_Variance_Held__{}__{}.png'.format(Output_Id,it), bbox_inches='tight')

        print('Variation maintained: %.2f' % variacao_percentual_pca.sum())
        print('                  ')

        if (Chose != 'Test'):
            features_reduzidas = pca.transform(features_padronizadas)
            print('Filtered Features')
            print('-' * 20)
            print(np.size(features_padronizadas,0))
            print(np.size(features_padronizadas,1))
            print('-' * 20)
            print('Reduced Features')
            print('-' * 20)
            print(np.size(features_reduzidas,0))
            print(np.size(features_reduzidas,1))

            if (Chose != 'Test'):
                
                ### Análise de atributos ###


                eigen_matrix = np.array(pca.components_)
                eigen_matrix = pow((pow(eigen_matrix,2)),0.5) #invertendo valores negativos

                for i in range (eigen_matrix.shape[0]):

                    LineSum = sum(eigen_matrix[i,:])
                    for j in range (eigen_matrix.shape[1]):
                        eigen_matrix[i,j] = ((eigen_matrix[i,j]*100)/LineSum)


                if Chose == 'Specific':
                ### Análise Expecífica ###

                    fig = plt.figure(figsize=[16,int(8*N_PCs)])

                    fig.suptitle('Contribution percentage per PC', fontsize=16)

                    ax = fig.subplots(int(N_PCs),1)

                    for i in range (int(N_PCs)):

                        s = eigen_matrix[i,:]

                        ax[i].bar(x=range(0,(eigen_matrix.shape[1])),height=s)
                        ax[i].set(xlabel='Features', ylabel='Contribution Percentage', title = 'PC ' + str(i+1))
                        ax[i].grid()


                    # Hide x labels and tick labels for top plots and y ticks for right plots.
                    for axs in ax.flat:
                        axs.label_outer()

                    plt.show()
                    fig.savefig('Contribution_Percentage_Per_PC_{}.png'.format(Output_Id), bbox_inches='tight')

                if (Chose == 'Analytics'):
                    ### Análise Geral ###

                    weighted_contribution = np.zeros((2,eigen_matrix.shape[1]))

                    for i in range (eigen_matrix.shape[1]):
                        NumeratorSum = 0
                        for j in range (N_PCs):
                            NumeratorSum += eigen_matrix[j,i] * variacao_percentual_pca[j]

                        weighted_contribution[0,i] = NumeratorSum / sum(variacao_percentual_pca)

                    df_weighted_contribution = pd.DataFrame(weighted_contribution,columns=selected_names)
                    df_weighted_contribution = df_weighted_contribution.drop([1])                    
                    df_weighted_contribution = df_weighted_contribution.sort_values(by=0, axis=1, ascending=False)
                    
                    
                    #pd.set_option('display.max_rows', len(df_weighted_contribution))
                    #print(type(df_weighted_contribution))
                    #print(df_weighted_contribution.head())
                    #pd.reset_option('display.max_rows')

                    #Creating Separated Data Frames por Sensors and Features Contribution 

                    sensors_names = [None] * int(df_weighted_contribution.shape[1])
                    features_names = [None] * int(df_weighted_contribution.shape[1])
                    general_features = [None] * int(df_weighted_contribution.shape[1])


                    for i, names in zip(range (df_weighted_contribution.shape[1]), df_weighted_contribution.columns):

                        c = '__'
                        words = names.split(c)
                        
                        sensors_names[i] = words[0]
                        general_features[i]= words[1]
                        features_names[i] = c.join(words[1:])

                        #print(names)
                        #print(words)
                        #print(sensors_names[i])
                        #print(features_names[i])
                        #print(50*'-')

                    
                    unique_sensors_names = np.ndarray.tolist(np.unique(np.array(sensors_names))) 
                    unique_general_feature = np.ndarray.tolist(np.unique(np.array(general_features))) 
                    unique_features_names = np.ndarray.tolist(np.unique(np.array(features_names)))
                    sensors_contribution = pd.DataFrame (np.zeros((2,np.shape(unique_sensors_names)[0])), columns=unique_sensors_names)
                    general_features_contribution = pd.DataFrame (np.zeros((2,np.shape(unique_general_feature)[0])), columns=unique_general_feature)
                    features_contribution = pd.DataFrame (np.zeros((2,np.shape(unique_features_names)[0])), columns=unique_features_names)
                    sensors_contribution = sensors_contribution.drop([1])
                    general_features_contribution = general_features_contribution.drop([1])
                    features_contribution = features_contribution.drop([1])
                    
                    
                    # For the output Formating
                    
                    """
                    unique_sensors_names = np.ndarray.tolist(np.unique(np.array(sensors_names))) 
                    unique_features_names = np.ndarray.tolist(np.unique(np.array(features_names)))
                    sensor_dt = np.transpose(np.vstack((unique_sensors_names,np.asarray(np.zeros(np.shape(unique_sensors_names)[0]),object))))
                    feature_dt = np.transpose(np.vstack((unique_features_names,np.asarray(np.zeros(np.shape(unique_features_names)[0]),object))))
                    sensors_contribution = pd.DataFrame(sensor_dt,columns = ['Sensor','Contribution'])
                    features_contribution = pd.DataFrame(feature_dt,columns = ['Feature','Contribution'])
                    """
                    #print(sensors_contribution.head())
                    #print(features_contribution.head())

                    #Creating dictionaries form Data Frame orientation
                    
                    """
                    Creates a mapping from kind names to fc_parameters objects
                    (which are itself mappings from feature calculators to settings)
                    to extract only the features contained in the columns.
                    To do so, for every feature name in columns this method

                    1. split the column name into col, feature, params part
                    2. decide which feature we are dealing with (aggregate with/without params or apply)
                    3. add it to the new name_to_function dict
                    4. set up the params

                    :param columns: containing the feature names
                    :type columns: list of str
                    :param columns_to_ignore: columns which do not contain tsfresh feature names
                    :type columns_to_ignore: list of str

                    :return: The kind_to_fc_parameters object ready to be used in the extract_features function.
                    :rtype: dict
                    """

                    weighted_contribution_dic = {}

                    for col in df_weighted_contribution.columns:

                        # Split according to our separator into <col_name>, <feature_name>, <feature_params>
                        parts = col.split('__')
                        n_parts = len(parts)

                        if n_parts == 1:
                            raise ValueError("Splitting of columnname {} resulted in only one part.".format(col))

                        kind = parts[0]
                        feature = c.join(parts[1:])
                        feature_name = parts[1]

                        if kind not in weighted_contribution_dic:
                            weighted_contribution_dic[kind] = {}

                        if not hasattr(feature_calculators, feature_name):
                            raise ValueError("Unknown feature name {}".format(feature_name))
                            
                        sensors_contribution.loc[0,kind] += df_weighted_contribution.loc[0,col]
                        general_features_contribution.loc[0,feature_name] += df_weighted_contribution.loc[0,col]
                        features_contribution.loc[0,feature] += df_weighted_contribution.loc[0,col]
                        weighted_contribution_dic[kind][feature] = df_weighted_contribution.loc[0,col]
                    
                        
                    # End of the tsfresh stolen function

                    
                    """
                    sensors_dic = {}
                    for i in range(len(unique_sensors_names)):
                        sensors_dic[unique_sensors_names[i]] = i

                    features_dic = {}
                    for i in range(len(unique_features_names)):
                        features_dic[unique_features_names[i]] = i

                    #Suming the contibution for Sensors and Features

                    for i in range(df_weighted_contribution.shape[0]):

                        names = df_weighted_contribution.loc[i,'tsfresh_info']
                        c = '__'
                        words = names.split(c)           
                        S= words[0]
                        F= c.join(words[1:])

                        sensors_contribution.loc[sensors_dic[S],'Contribution'] += df_weighted_contribution.loc[i,'Contribution']
                        features_contribution.loc[features_dic[F],'Contribution'] += df_weighted_contribution.loc[i,'Contribution']

                    sensors_contribution = sensors_contribution.sort_values(by=['Contribution'], ascending=False)
                    features_contribution = features_contribution.sort_values(by=['Contribution'], ascending=False)
                    """
                    
                    features_contribution = features_contribution.sort_values(by=0, axis=1, ascending=False)
                    general_features_contribution = general_features_contribution.sort_values(by=0, axis=1, ascending=False)
                    
                    features_indexes = [x for x in range(1,(features_contribution.shape[0])+1)]
                    general_features_indexes = [x for x in range(1,(general_features_contribution.shape[0])+1)]

                    features_contribution.set_index(pd.Index(features_indexes))
                    general_features_contribution.set_index(pd.Index(general_features_indexes))
                    
                    sorted_sensors_contribution = sensors_contribution.values[0,:]
                    sorted_features_contribution = features_contribution.values[0,:]
                    sorted_general_features_contribution = general_features_contribution.values[0,:]

                    #Ploting Cntribution Sensors Results
                    
                    fig = plt.figure(figsize=[16,8])

                    #fig.suptitle('Sensors Weighted Contribution Percentage', fontsize=16)

                    ax = fig.subplots(1,1)

                    s = sorted_sensors_contribution[:]

                    ax.bar(x=['Voltage','Current'],height=s)
                    plt.ylabel('Relevance Percentage',fontsize = 20)
                    plt.xlabel('Sensors',fontsize = 20)
                    plt.tick_params(axis='x', labelsize=16)
                    plt.tick_params(axis='y', labelsize=18)
                    ax.grid()

                    plt.show()
                    fig.savefig('Sensor_Weighted_Contribution_Percentage_{}.png'.format(Output_Id), bbox_inches='tight')

                    #Ploting Cntribution Features Results

                    fig = plt.figure(figsize=[16,8])

                    #fig.suptitle('Features Weighted Contribution Percentage', fontsize=16)

                    ax = fig.subplots(1,1)

                    s = sorted_features_contribution[:]

                    ax.bar(x=range(0,(sorted_features_contribution.shape[0])),height=s)
                    plt.ylabel('Relevance Percentage',fontsize = 20)
                    plt.xlabel('Features',fontsize = 20)
                    plt.tick_params(axis='x', labelsize=16)
                    plt.tick_params(axis='y', labelsize=18)
                    ax.xaxis.set_major_locator(plt.MultipleLocator(50))
                    ax.xaxis.set_minor_locator(plt.MultipleLocator(50))
                    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
                    ax.grid()

                    plt.show()
                    fig.savefig('Features_Weighted_Contribution_Percentage_{}.png'.format(Output_Id), bbox_inches='tight')


                    ### Análise Geral para os 20 melhores atributos completos ###

                    fig = plt.figure(figsize=[16,8])

                    #fig.suptitle('Best Features Weighted Contribution Percentage', fontsize=16)

                    #print('Porcentagem de pertinência: ', np.sum(sorted_features_contribution[0:140]))
                    #print('Number of Selected Features: ', sorted_features_contribution.shape[0])

                    ax = fig.subplots(1,1)

                    s = sorted_features_contribution[0:20]

                    ax.bar(x=['X' + str(x) for x in range(1,(20+1))],height=s)
                    plt.ylabel('Relevance Percentage',fontsize = 20)
                    plt.xlabel('Features',fontsize = 20)
                    plt.tick_params(axis='x', labelsize=16)
                    plt.tick_params(axis='y', labelsize=18)
                    ax.grid()

                    plt.show()
                    fig.savefig('{}th_Best_Features_Weighted_Contribution_Percentage_{}.png'.format(20,Output_Id), bbox_inches='tight')

                    ### Análise Geral para os 20 melhores atributos gerais ###

                    fig = plt.figure(figsize=[16,8])

                    #fig.suptitle('Best Features Weighted Contribution Percentage', fontsize=16)

                    #print('Porcentagem de pertinência: ', np.sum(sorted_features_contribution[0:140]))
                    #print('Number of Selected Features: ', sorted_features_contribution.shape[0])

                    ax = fig.subplots(1,1)

                    s = sorted_features_contribution[0:20]

                    ax.bar(x=['X' + str(x) for x in range(1,(20+1))],height=s)
                    plt.ylabel('Relevance Percentage',fontsize = 20)
                    plt.xlabel('Features',fontsize = 20)
                    plt.tick_params(axis='x', labelsize=16)
                    plt.tick_params(axis='y', labelsize=18)
                    ax.grid()
                    ax.set_ylim([s[-1]-0.05,s[0]+0.05])
                    plt.show()
                    fig.savefig('{}th_Best_Features_Weighted_Contribution_Percentage_{}_zoom.png'.format(20,Output_Id), bbox_inches='tight')


                    #Ploting the data of the most relevant sensor with the best features

                    sensors_contribution.values[:,0]

                    name_1 = df_weighted_contribution.columns[0]
                    name_2 = df_weighted_contribution.columns[1]
                    name_3 = df_weighted_contribution.columns[2]


                    #pd.set_option('display.max_columns', len(features))
                    #print(features)
                    #pd.reset_option('display.max_columns')

                    x = features.loc[:,name_1].values
                    y = features.loc[:,name_2].values
                    z = features.loc[:,name_3].values
                    data_saida = np.array([x, y, z]).T
                    
                    np.savetxt('atributos.csv', data_saida, delimiter=',')

                    x = scale(x,-1,1)
                    y = scale(y,-1,1)
                    z = scale(z,-1,1)

                    x_bom=[]
                    x_ruim=[]
                    y_bom=[]
                    y_ruim=[]
                    z_bom=[]
                    z_ruim=[]
                    
                    for i in range(len(Target)):
                        if Target[i] == 0:
                            x_bom.append(x[i])
                            y_bom.append(y[i])
                            z_bom.append(z[i])
                        if Target[i] == 1:
                            x_ruim.append(x[i])
                            y_ruim.append(y[i])
                            z_ruim.append(z[i])
                            
                    os.chdir( base_path )
                            
                    #np.savetxt('x_bom.csv', x_bom, delimiter=',')
                    #np.savetxt('x_ruim.csv', x_ruim, delimiter=',')
                    
                    os.chdir( PCA_Figures_path )

                    fig = plt.figure(figsize=[14,10])
                    ax = fig.add_subplot(111, projection='3d')

                    ax.scatter(x_bom, y_bom, z_bom, c = 'blue' )
                    ax.scatter(x_ruim, y_ruim, z_ruim, c = 'red' )

                    plt.ylabel('X2',fontsize = 20,labelpad=18)
                    plt.xlabel('X1',fontsize = 20, labelpad=18)
                    ax.set_zlabel('X3', fontsize = 20, labelpad=12)
                    plt.tick_params(axis='x', labelsize=16)
                    plt.tick_params(axis='y', labelsize=16)
                    plt.tick_params(axis='z', labelsize=16)
                    ax.grid()
                    red_patch = mpatches.Patch(color='red', label='Non-Funcional Tools')
                    blue_patch = mpatches.Patch(color='blue', label='Funcional Tools')
                    plt.legend(handles=[red_patch,blue_patch],fontsize = 20)
                    #plt.show()
                    fig.savefig('ScatterPlot_PCA_{}.png'.format(Output_Id), bbox_inches='tight')
                    
                    # -------------------------------------------
                    fig = plt.figure(figsize=[21,7])
                    ax = fig.subplots(1,3)
                    
                    ax[0].scatter(x_bom, y_bom, c = 'blue' )
                    ax[0].scatter(x_ruim, y_ruim, c = 'red' )
                    ax[0].set_xlabel('X1',fontsize = 20)
                    ax[0].set_ylabel('X2',fontsize = 20)
                    ax[0].grid()
                    
                    ax[1].scatter(x_bom, z_bom, c = 'blue' )
                    ax[1].scatter(x_ruim, z_ruim, c = 'red' )
                    ax[1].set_xlabel('X1',fontsize = 20)
                    ax[1].set_ylabel('X3',fontsize = 20)
                    ax[1].grid()
                    
                    ax[2].scatter(y_bom, z_bom, c = 'blue' )
                    ax[2].scatter(y_ruim, z_ruim, c = 'red' )
                    ax[2].set_xlabel('X2',fontsize = 20,)
                    ax[2].set_ylabel('X3',fontsize = 20)
                    ax[2].grid()
                    
                    #plt.show()
                    fig.savefig('X1X2X3_{}.png'.format(Output_Id), bbox_inches='tight')
                    
                    
                    # -------------------------------------------
                    # Now change to PCA Analyses directory

                    os.chdir( PCA_Analyses_path )

                    general_features_contribution.to_csv('unique_features_used_{}.csv'.format(Output_Id),index = False)
                    sensors_contribution.to_csv('sensors_weighted_contribution_{}.csv'.format(Output_Id), index=True)
                    features_contribution.to_csv('features_weighted_contribution_{}.csv'.format(Output_Id), index=True)

            # Now change to PCA Analyses directory

            
            # -------------------------------------------
            
            x = features_reduzidas[:,0]
            y = features_reduzidas[:,1]
            z = features_reduzidas[:,2]

            x_bom=[]
            x_ruim=[]
            y_bom=[]
            y_ruim=[]
            z_bom=[]
            z_ruim=[]
                    
            for i in range(len(Target)):
                if Target[i] == 0:
                    x_bom.append(x[i])
                    y_bom.append(y[i])
                    z_bom.append(z[i])
                if Target[i] == 1:
                    x_ruim.append(x[i])
                    y_ruim.append(y[i])
                    z_ruim.append(z[i])
                            
            os.chdir( PCA_Figures_path )

            fig = plt.figure(figsize=[14,10])
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(x_bom, y_bom, z_bom, c = 'blue' )
            ax.scatter(x_ruim, y_ruim, z_ruim, c = 'red' )

            plt.ylabel('PC2',fontsize = 20,labelpad=18)
            plt.xlabel('PC1',fontsize = 20, labelpad=18)
            ax.set_zlabel('PC3', fontsize = 20, labelpad=12)
            plt.tick_params(axis='x', labelsize=16)
            plt.tick_params(axis='y', labelsize=16)
            plt.tick_params(axis='z', labelsize=16)
            ax.grid()
            red_patch = mpatches.Patch(color='red', label='Non-Funcional Tools')
            blue_patch = mpatches.Patch(color='blue', label='Funcional Tools')
            plt.legend(handles=[red_patch,blue_patch],fontsize = 20)
            #plt.show()
            fig.savefig('ScatterPlot_features__{}__{}.png'.format(Output_Id,it), bbox_inches='tight')
            
            # -------------------------------------------
            fig = plt.figure(figsize=[21,7])
            ax = fig.subplots(1,3)
                    
            ax[0].scatter(x_bom, y_bom, c = 'blue' )
            ax[0].scatter(x_ruim, y_ruim, c = 'red' )
            ax[0].set_xlabel('PC1',fontsize = 20)
            ax[0].set_ylabel('PC2',fontsize = 20)
            ax[0].grid()
                    
            ax[1].scatter(x_bom, z_bom, c = 'blue' )
            ax[1].scatter(x_ruim, z_ruim, c = 'red' )
            ax[1].set_xlabel('PC1',fontsize = 20)
            ax[1].set_ylabel('PC3',fontsize = 20)
            ax[1].grid()
                    
            ax[2].scatter(y_bom, z_bom, c = 'blue' )
            ax[2].scatter(y_ruim, z_ruim, c = 'red' )
            ax[2].set_xlabel('PC2',fontsize = 20,)
            ax[2].set_ylabel('PC3',fontsize = 20)
            ax[2].grid()
                    
            #plt.show()
            fig.savefig('PC1PC2PC3__{}__{}.png'.format(Output_Id,it), bbox_inches='tight')
                    
                    
            # -------------------------------------------
            # -------------------------------------------
            
            os.chdir( PCA_Analyses_path )
            np.savetxt("features_reduzidas_" + str(Output_Id) + ".csv", features_reduzidas, delimiter=',')

            Output = {'ReducedFeatures': features_reduzidas,
                      'ID': Output_Id} 
        elif (Chose == 'Test'): 

            Output = {'ID': Output_Id}
        
    # Now change back to base directory

    os.chdir( base_path )

    return Output

def PCA_projection (features,N_PCs):
    ''' Function for ONLINE mode
    This function project the data into a trained PCA.
    
    Parameters:
    ------
    features: dataframe 
        #############################################################
    
    Returns: 
    -------
    dataframe
        contain the output data of PCA, i.e., the dataset with Principal Componentes projected by PCA
    
    '''
    loaded_scaler = pickle.load(open('Kernel/pca_scaler.sav', 'rb'))
    features_padronizadas = loaded_scaler.transform(features)
    #centralizar os dados e colocá-los com desvioPadrão=1
    #scaler = StandardScaler().fit(features)
    #features_padronizadas = scaler.transform(features)

    pca= PCA(n_components = N_PCs)
    pca.fit(features_padronizadas)
    
    variacao_percentual_pca = np.round(pca.explained_variance_ratio_ * 100, decimals = 2)

    print('Variation maintained: %.2f' % variacao_percentual_pca.sum())
    print('                  ')


    features_reduzidas = pca.transform(features_padronizadas)

    """
    # load the model from disk
    loaded_pca = pickle.load(open('Kernel/pca.sav', 'rb'))
    
    scaler = StandardScaler().fit(features)
    features_padronizadas = scaler.transform(features)

    features_padronizadas = scaler.transform(features)
    features_reduzidas = loaded_pca.transform(features_padronizadas)
    """
    
    return features_reduzidas

class cpu_usage(threading.Thread):### Thread to calculate duration and mean cpu percente usage in a SODA classifier
    def __init__(self):
        threading.Thread.__init__(self)
        self.control = True
        
    def run(self):
        cpu = []
        t_inicial = time.time()
        while self.control:
            cpu.append(cpu_percent(interval=1, percpu=True))
        t_final = time.time()
        self.deltatime = t_final - t_inicial
        self.mean_cpu = np.mean(cpu)
        
    def stop(self):
        self.control = False
        
    def join(self):
        threading.Thread.join(self)
        return self.deltatime, self.mean_cpu

def grid_set(data, N): #SODA process
    _ , W = data.shape
    AvD1 = data.mean(0)
    X1 = np.mean(np.sum(np.power(data,2),axis=1))
    grid_trad = np.sqrt(2*(X1 - AvD1*AvD1.T))/N
    Xnorm = np.sqrt(np.sum(np.power(data,2),axis=1))
    aux = Xnorm
    for _ in range(W-1):
        aux = np.insert(aux,0,Xnorm.T,axis=1)
    data = data / aux
    seq = np.argwhere(np.isnan(data))
    if tuple(seq[::]): data[tuple(seq[::])] = 1
    AvD2 = data.mean(0)
    grid_angl = np.sqrt(1-AvD2*AvD2.T)/N
    return X1, AvD1, AvD2, grid_trad, grid_angl

def pi_calculator(Uniquesample, mode):#SODA process
    UN, W = Uniquesample.shape
    if mode == 'euclidean' or mode == 'mahalanobis' or mode == 'cityblock' or mode == 'chebyshev' or mode == 'canberra':
        AA1 = Uniquesample.mean(0)
        X1 = sum(sum(np.power(Uniquesample,2)))/UN
        DT1 = X1 - sum(np.power(AA1,2))
        aux = []
        for i in range(UN): aux.append(AA1)
        aux2 = [Uniquesample[i]-aux[i] for i in range(UN)]
        uspi = np.sum(np.power(aux2,2),axis=1)+DT1
    
    if mode == 'minkowski':
        AA1 = Uniquesample.mean(0)
        X1 = sum(sum(np.power(Uniquesample,2)))/UN
        DT1 = X1 - sum(np.power(AA1,2))
        aux = np.matrix(AA1)
        for i in range(UN-1): aux = np.insert(aux,0,AA1,axis=0)
        aux = np.array(aux)
        uspi = np.sum(np.power(cdist(Uniquesample, aux, mode, p=1.5),2),1)+DT1
    
    if mode == 'cosine':
        Xnorm = np.matrix(np.sqrt(np.sum(np.power(Uniquesample,2),axis=1))).T
        aux2 = Xnorm
        for i in range(W-1):
            aux2 = np.insert(aux2,0,Xnorm.T,axis=1)
        Uniquesample1 = Uniquesample / aux2
        AA2 = np.mean(Uniquesample1,0)
        X2 = 1
        DT2 = X2 - np.sum(np.power(AA2,2))
        aux = []
        for i in range(UN): aux.append(AA2)
        aux2 = [Uniquesample1[i]-aux[i] for i in range(UN)]
        uspi = np.sum(np.sum(np.power(aux2,2),axis=1),axis=1)+DT2
        
    return uspi

def Globaldensity_Calculator(data, distancetype):#SODA process
    
    Uniquesample, J, K = np.unique(data, axis=0, return_index=True, return_inverse=True)
    Frequency, _ = np.histogram(K,bins=len(J))
    uspi1 = pi_calculator(Uniquesample, distancetype)
    sum_uspi1 = sum(uspi1)
    Density_1 = uspi1 / sum_uspi1
    uspi2 = pi_calculator(Uniquesample, 'cosine')
    sum_uspi2 = sum(uspi2)
    Density_2 = uspi2 / sum_uspi2
    
    GD = (Density_2+Density_1) * Frequency

    index = GD.argsort()[::-1]
    GD = GD[index]
    Uniquesample = Uniquesample[index]
    Frequency = Frequency[index]
 
    return GD, Uniquesample, Frequency

def chessboard_division(Uniquesample, MMtypicality, interval1, interval2, distancetype):#SODA process
    L, W = Uniquesample.shape
    if distancetype == 'euclidean':
        W = 1
    BOX = [Uniquesample[k] for k in range(W)]
    BOX_miu = [Uniquesample[k] for k in range(W)]
    BOX_S = [1]*W
    BOX_X = [sum(Uniquesample[k]**2) for k in range(W)]
    NB = W
    BOXMT = [MMtypicality[k] for k in range(W)]
    
    for i in range(W,L):
        if distancetype == 'minkowski':
            a = cdist(Uniquesample[i].reshape(1,-1), BOX_miu, metric=distancetype, p=1.5)
        else:
            a = cdist(Uniquesample[i].reshape(1,-1), BOX_miu, metric=distancetype)
        
        b = np.sqrt(cdist(Uniquesample[i].reshape(1,-1), BOX_miu, metric='cosine'))
        distance = np.array([a[0],b[0]]).T
        SQ = []
        for j,d in enumerate(distance):
            if d[0] < interval1 and d[1] < interval2:
                SQ.append(j)
        #SQ = np.argwhere(distance[::,0]<interval1 and (distance[::,1]<interval2))
        COUNT = len(SQ)
        if COUNT == 0:
            BOX.append(Uniquesample[i])
            NB = NB + 1
            BOX_S.append(1)
            BOX_miu.append(Uniquesample[i])
            BOX_X.append(sum(Uniquesample[i]**2))
            BOXMT.append(MMtypicality[i])
        if COUNT >= 1:
            DIS = distance[SQ[::],0]/interval1 + distance[SQ[::],1]/interval2 # pylint: disable=E1136  # pylint/issues/3139
            b = np.argmin(DIS)
            BOX_S[SQ[b]] = BOX_S[SQ[b]] + 1
            BOX_miu[SQ[b]] = (BOX_S[SQ[b]]-1)/BOX_S[SQ[b]]*BOX_miu[SQ[b]] + Uniquesample[i]/BOX_S[SQ[b]]
            BOX_X[SQ[b]] = (BOX_S[SQ[b]]-1)/BOX_S[SQ[b]]*BOX_X[SQ[b]] + sum(Uniquesample[i]**2)/BOX_S[SQ[b]]
            BOXMT[SQ[b]] = BOXMT[SQ[b]] + MMtypicality[i]


    return BOX, BOX_miu, BOX_X, BOX_S, BOXMT, NB

def ChessBoard_PeakIdentification(BOX_miu,BOXMT,NB,Internval1,Internval2, distancetype):#SODA process
    Centers = []
    n = 2
    ModeNumber = 0
           
    if distancetype == 'minkowski':
        distance1 = squareform(pdist(BOX_miu,metric=distancetype, p=1.5))
    else:
        distance1 = squareform(pdist(BOX_miu,metric=distancetype))        

    distance2 = np.sqrt(squareform(pdist(BOX_miu,metric='cosine')))
      
    for i in range(NB):
        seq = []
        for j,(d1,d2) in enumerate(zip(distance1[i],distance2[i])):
            if d1 < n*Internval1 and d2 < n*Internval2:
                seq.append(j)
        Chessblocak_typicality = [BOXMT[j] for j in seq]

        if max(Chessblocak_typicality) == BOXMT[i]:
            Centers.append(BOX_miu[i])
            ModeNumber = ModeNumber + 1

    return Centers, ModeNumber

def cloud_member_recruitment(ModelNumber,Center_samples,Uniquesample,grid_trad,grid_angl, distancetype):#SODA process
    L, W = Uniquesample.shape
    Membership = np.zeros((L,ModelNumber))
    Members = np.zeros((L,ModelNumber*W))
    Count = []
    
    if distancetype == 'minkowski':
        distance1 = cdist(Uniquesample,Center_samples, metric=distancetype, p=1.5)/grid_trad
    else:
        distance1 = cdist(Uniquesample,Center_samples, metric=distancetype)/grid_trad

    distance2 = np.sqrt(cdist(Uniquesample, Center_samples, metric='cosine'))/grid_angl
    distance3 = distance1 + distance2
    B = distance3.argmin(1)

    for i in range(ModelNumber):
        seq = []
        for j,b in enumerate(B):
            if b == i:
                seq.append(j)
        Count.append(len(seq))
        Membership[:Count[i]:,i] = seq
        Members[:Count[i]:,W*i:W*(i+1)] = [Uniquesample[j] for j in seq]
    MemberNumber = Count
    
    #Converte a matriz para vetor E SOMA +1 PARA NAO TER CONJUNTO 0'
    B = B.A1
    B = [x+1 for x in B]
    return Members,MemberNumber,Membership,B

def SelfOrganisedDirectionAwareDataPartitioning(Input, Mode='Offline'):#SODA process
    ''' Main function of SODA
    
    Parameters:
    ------
    Input: dictionary, with the following items
        'GridSize': float
            current granularity value
        'StaticData': np.matrix
            data
        'DistanceType': str
            current magnitude distance metrics, can be
            ['euclidean', 'mahalanobis', 'cityblock', 'chebyshev', 'minkowski', 'canberra']
    
    Mode: str
        SODA Algorithm mode, can be ['Offline', 'Evolvig']
        (default = 'Offline')
    
    Returns: 
    -------
    dictionary, with the following items
        'C': list
            list of center coordenates
        'IDX': list
            list of the corresponding index of the data cloud to which each event belongs
        'SystemParams': dictionary, with the following items
            'BOX': list
            'BOX_miu': list
            'BOX_S': list
            'NB': int
            'XM': float
            'L': int
                number of events
            'AvM': np.matrix
            'AvA': np.matrix
            'GridSize': int
                current granularity value
        'DistanceType': str 
            current magnitude distance metrics, can be
            ['euclidean', 'mahalanobis', 'cityblock', 'chebyshev', 'minkowski', 'canberra']
    '''
    
    
    if Mode == 'Offline':
        data = Input['StaticData']

        L = data.shape[0]
        N = Input['GridSize']
        distancetype = Input['DistanceType']
        X1, AvD1, AvD2, grid_trad, grid_angl = grid_set(data,N)
        GD, Uniquesample, Frequency = Globaldensity_Calculator(data, distancetype)

        BOX,BOX_miu,BOX_X,BOX_S,BOXMT,NB = chessboard_division(Uniquesample,GD,grid_trad,grid_angl, distancetype)
        Center,ModeNumber = ChessBoard_PeakIdentification(BOX_miu,BOXMT,NB,grid_trad,grid_angl, distancetype)
        Members,Membernumber,Membership,IDX = cloud_member_recruitment(ModeNumber,Center,data,grid_trad,grid_angl, 
                                                                       distancetype)
        
        Boxparameter = {'BOX': BOX,
                'BOX_miu': BOX_miu,
                'BOX_S': BOX_S,
                'NB': NB,
                'XM': X1,
                'L': L,
                'AvM': AvD1,
                'AvA': AvD2,
                'GridSize': N}
        
    if Mode == 'Evolving':
        #TODO
        print(Mode)

    Output = {'C': Center,
              'IDX': IDX,
              'SystemParams': Boxparameter,
              'DistanceType': distancetype}
           
    return Output

def SODA (ReducedFeatures, min_granularity, max_granularity, pace):#SODA
    ''' Start of SODA
    
    Parameters:
    ------
    ReducedFeatures : dictionary, with the following items
        'ReducedFeatures': np.array
            contain the output data of PCA, i.e., the dataset with Principal Componentes projected by PCA
        'ID': int
            identifier for the dataset
    
    min_granularity: float
        first value of granularity for SODA algorithm
    
    max_granularity: float
        final value of granularity for SODA algorithm
        
    pace: float
        increase of granularity for SODA algorithm
    
    Returns: 
    -------
    dictionary, with the following items
        'granularity_i': out['idx']
            Labels of the SODA data clouds
        ...
        'granularity_n': 

        'ID':int
            identifier for the dataset
    '''
    DataSetID = ReducedFeatures['ID']
    data = ReducedFeatures['ReducedFeatures']
    data = np.matrix(data)

    
    #### Looping SODA within the chosen granularities and distances ####
    Output = {}

    for g in np.arange(int(min_granularity), int (max_granularity + pace), pace):

        print('Processing granularity %d'%g)

        Input = {'GridSize':g, 'StaticData':data, 'DistanceType': 'euclidean'}
        
        out = SelfOrganisedDirectionAwareDataPartitioning(Input,'Offline')

        Output['granularity_' + str(g)] = out['IDX']

    Output['ID'] = DataSetID
    
    return Output

def GroupingAlgorithm (SODA_parameters): #Grouping Algorithm
    print('             ')
    print('Grouping Algorithm Control Output')
    print('----------------------------------')
    
    ####   imput data    ####
    DataSetID = SODA_parameters['ID']
    
    with open('Kernel/final_target_'+str(DataSetID)+'.pkl', 'rb') as f:
        y_original = pickle.load(f)

    Output = {}

    for gra in SODA_parameters:
        if gra  != 'ID':
            SodaOutput = SODA_parameters[gra]
            
            #### Program Matrix's and Variables ####

            define_percent = 50
            n_DA_planes = np.max(SodaOutput)
            Percent = np.zeros((int(n_DA_planes),3))
            n_IDs_per_gp = np.zeros((int(n_DA_planes),2))
            n_tot_Id_per_DA = np.zeros((int(n_DA_planes),1))
            decision = np.zeros(int(n_DA_planes))
            selected_samples = np.zeros(2)
            n_gp0 = 0
            n_gp1 = 0
            k = 0

            #### Definition Percentage Calculation #####

            for i in range(y_original.shape[0]):

                if y_original[i] == 0:
                    n_IDs_per_gp [int(SodaOutput[i]-1),0] += 1 
                else:
                    n_IDs_per_gp [int(SodaOutput[i]-1),1] += 1 

                n_tot_Id_per_DA [int(SodaOutput[i]-1)] += 1 


            for i in range(int(n_DA_planes)):

                Percent[i,0] = (n_IDs_per_gp[i,0] / n_tot_Id_per_DA[i]) * 100
                Percent[i,1] = (n_IDs_per_gp[i,1] / n_tot_Id_per_DA[i]) * 100
            
            #### Using Definition Percentage as Decision Parameter ####

            for i in range(Percent.shape[0]): # pylint: disable=E1136  # pylint/issues/3139

                if (Percent[i,0] > define_percent):
                    n_gp0 += 1
                    decision[i] = 0
                else:
                    n_gp1 += 1
                    decision[i] = 1
                  
            #### Defining labels

            ClassifiersLabel = []

            for i in range (len(SodaOutput)):
                ClassifiersLabel.append(decision[int (SodaOutput[i]-1)])
            
            Output[gra] = ClassifiersLabel

            ### Printig Analitics results
            
            print(gra)
            print('Number of data clouds: %d' % n_DA_planes)
            print('Number of good tools groups: %d' % n_gp0)
            print('Number of worn tools groups: %d' % n_gp1)
            print('Number of samples: %d' % int(len(SodaOutput)))
            print('---------------------------------------------------')

            
            # Saving analysis result
            
            Grouping_Analyse = open("Kernel/Grouping_Analyse__" + str(DataSetID) + '__' + str(gra) + "__.txt","a+")
            Grouping_Analyse.write(gra)
            Grouping_Analyse.write('\nNumber of data clouds: %d\n' % n_DA_planes)
            Grouping_Analyse.write('Number of good tools groups: %d\n' % n_gp0)
            Grouping_Analyse.write('Number of worn tools groups: %d\n' % n_gp1)
            Grouping_Analyse.write('Number of samples: %d\n' % len(SodaOutput))
            Grouping_Analyse.write('---------------------------------------------------')
            
            Grouping_Analyse.close()
    
    return Output

def Classification (ClassificationPar, min_granularity,max_granularity,n_a): #Classifiers
    
    #Changing Work Folder
    add_path1 = "/Classification/"
    add_path2 = "/Kernel/"
    add_path3 = "/.Recovery/"
    base_path = os.getcwd()
    working_path = os.getcwd()
    Classification_path = working_path + add_path1
    Kernel_path = working_path + add_path2
    Recovery_path = working_path + add_path3

    # Change to Kernel directory
    os.chdir(Kernel_path)

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]


    classifiers = [
        KNeighborsClassifier(3),
        SVC(gamma='scale'),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100),
        MLPClassifier(alpha=1,max_iter=500),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]
    
    Output_ID = ClassificationPar['ID']
    distances = ClassificationPar['Distances']
    pace = ClassificationPar['Pace']
    gra = np.arange(min_granularity,max_granularity,pace)

    for d in distances:
        for g in gra:
            try:
                # Now change to Kernel directory

                os.chdir( Kernel_path )

                # preprocess dataset, split into training and test part
                Accuracy = np.zeros((n_a, len(classifiers) + 1))
                Precision = np.zeros((n_a, len(classifiers) + 1))
                Recall = np.zeros((n_a, len(classifiers) + 1))
                F1 = np.zeros((n_a, len(classifiers) + 1))
                s = str (int(ClassificationPar['Percent'] )) + '_' + d + '_Labels_' + str(int(ClassificationPar['ID'])) + '_' + str("%.2f" % g) + '.npy'
                X = np.load('X_' + s)    
                y_soda = np.load('Y_' + s)
                y_original = np.load('Original_Y_' + s)

                #Loop into numbeer od samples
                for i in range(Accuracy.shape[0]): # pylint: disable=E1136  # pylint/issues/3139
                    k = 0
                    # iterate over classifiers

                    X_train, X_test, y_train_soda, y_test_soda, y_train_original, y_test_original = train_test_split(X, y_soda, y_original, test_size=.4)

                    for name, clf in zip(names, classifiers):

                        clf.fit(X_train, y_train_soda)
                        pickle.dump(clf, open('model.sav', 'wb'))
                        score = clf.score(X_test, y_test_original)
                        y_predict = list(clf.predict(X_test))
                        Accuracy[i,k] = score*100
                        Precision[i,k] = precision_score(y_test_original, y_predict, zero_division=0)*100
                        Recall[i,k] = recall_score(y_test_original, y_predict)*100
                        F1[i,k] = f1_score(y_test_original, y_predict)*100
                        k +=1
                        ClassifiersLabel = list(clf.predict(X_test))
                            
                results = []
                latex_results = []

                #Calculinng Mean and Std. Derivation 
                for i in range(len(names)):
                    results.append(['{:.2f} \u00B1 {:.2f}%'.format(np.mean(Accuracy[:,i]),np.std(Accuracy[:,i])),
                                    '{:.2f} \u00B1 {:.2f}%'.format(np.mean(Precision[:,i]),np.std(Precision[:,i])),
                                    '{:.2f} \u00B1 {:.2f}%'.format(np.mean(Recall[:,i]),np.std(Recall[:,i])),
                                    '{:.2f} \u00B1 {:.2f}%'.format(np.mean(F1[:,i]),np.std(F1[:,i]))])
                    
                    latex_results.append(['{:.2f} $\pm$ {:.2f} &'.format(np.mean(Accuracy[:,i]),np.std(Accuracy[:,i])),
                                    '{:.2f} $\pm$ {:.2f} &'.format(np.mean(Precision[:,i]),np.std(Precision[:,i])),
                                    '{:.2f} $\pm$ {:.2f} &'.format(np.mean(Recall[:,i]),np.std(Recall[:,i])),
                                    '{:.2f} $\pm$ {:.2f}'.format(np.mean(F1[:,i]),np.std(F1[:,i]))])

                # Now change to Grouping Analyses directory

                os.chdir( Classification_path )
                

                results = pd.DataFrame(results, columns=['Accuracy [%]', 'Precision [%]', 'Recall [%]', 'F1 [%]'], index=names)       
                results.to_csv(("Classification_result_" + s) )

                latex_results = pd.DataFrame(latex_results, columns=['Accuracy [%]', 'Precision [%]', 'Recall [%]', 'F1 [%]'], index=names)       
                latex_results.to_csv(("latex_result_" + s) )


                print('*** {} - {} - {:.2f}  ***'.format(ClassificationPar['ID'], d, g))
                print('-------------------------------------')
                print(results)
                print(' ')

            except:
                print('*** {} - {} - {:.2f}  ***'.format(Output_ID, d, g))
        
    # Now change to base directory

    os.chdir( base_path )

    return

def Classification_without_SODA (data,target,n_a): #Classifiers
    
    #Changing Work Folder
    add_path1 = "/Classification/"
    add_path2 = "/Kernel/"
    add_path3 = "/.Recovery/"
    base_path = os.getcwd()
    working_path = os.getcwd()
    Classification_path = working_path + add_path1
    Kernel_path = working_path + add_path2
    Recovery_path = working_path + add_path3

    # Change to Kernel directory
    os.chdir(Kernel_path)

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]


    classifiers = [
        KNeighborsClassifier(3),
        SVC(gamma='scale'),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100),
        MLPClassifier(alpha=1,max_iter=500),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    # Now change to Kernel directory

    os.chdir( Kernel_path )

    # preprocess dataset, split into training and test part
    Accuracy = np.zeros((n_a, len(classifiers) + 1))
    Precision = np.zeros((n_a, len(classifiers) + 1))
    Recall = np.zeros((n_a, len(classifiers) + 1))
    F1 = np.zeros((n_a, len(classifiers) + 1))

    #Loop into numbeer od samples
    for i in range(Accuracy.shape[0]): # pylint: disable=E1136  # pylint/issues/3139
        k = 0
        # iterate over classifiers

        X_train, X_test, y_train, y_test = train_test_split(data,target, test_size=.3)

        for name, clf in zip(names, classifiers):

            clf.fit(X_train, y_train)
            pickle.dump(clf, open('model.sav', 'wb'))
            score = clf.score(X_test, y_test)
            y_predict = list(clf.predict(X_test))
            Accuracy[i,k] = score*100
            Precision[i,k] = precision_score(y_test, y_predict, zero_division=0)*100
            Recall[i,k] = recall_score(y_test, y_predict)*100
            F1[i,k] = f1_score(y_test, y_predict)*100
            k +=1
            ClassifiersLabel = list(clf.predict(X_test))
                
    results = []
    latex_results = []

    #Calculinng Mean and Std. Derivation 
    for i in range(len(names)):
        results.append(['{:.2f} \u00B1 {:.2f}%'.format(np.mean(Accuracy[:,i]),np.std(Accuracy[:,i])),
                        '{:.2f} \u00B1 {:.2f}%'.format(np.mean(Precision[:,i]),np.std(Precision[:,i])),
                        '{:.2f} \u00B1 {:.2f}%'.format(np.mean(Recall[:,i]),np.std(Recall[:,i])),
                        '{:.2f} \u00B1 {:.2f}%'.format(np.mean(F1[:,i]),np.std(F1[:,i]))])
        
        latex_results.append(['{:.2f} $\pm$ {:.2f} &'.format(np.mean(Accuracy[:,i]),np.std(Accuracy[:,i])),
                        '{:.2f} $\pm$ {:.2f} &'.format(np.mean(Precision[:,i]),np.std(Precision[:,i])),
                        '{:.2f} $\pm$ {:.2f} &'.format(np.mean(Recall[:,i]),np.std(Recall[:,i])),
                        '{:.2f} $\pm$ {:.2f}'.format(np.mean(F1[:,i]),np.std(F1[:,i]))])

    # Now change to Grouping Analyses directory

    os.chdir( Classification_path )
    

    results = pd.DataFrame(results, columns=['Accuracy [%]', 'Precision [%]', 'Recall [%]', 'F1 [%]'], index=names)       
    results.to_csv(("Classification_result.csv") )

    latex_results = pd.DataFrame(latex_results, columns=['Accuracy [%]', 'Precision [%]', 'Recall [%]', 'F1 [%]'], index=names)       
    latex_results.to_csv(("latex_result.csv") )

    print('-------------------------------------')
    print(results)
    print(' ')


    return

def non_parametric_classification (X_train,X_test,GA_parameters,y_test): #Classifiers

    classifiers = [
        KNeighborsClassifier(3),
        SVC(gamma='scale'),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100),
        MLPClassifier(alpha=1,max_iter=500),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    Output = {}
    
    for gra in GA_parameters:

        print('-------------------------------------')
        print(gra)

        Output[gra] = {
        'Nearest Neighbors':{},
        'Linear SVM':{},
        'RBF SVM':{},
        'Gaussian Process':{},
        'Decision Tree':{},
        'Random Forest':{},
        'Neural Net':{},
        'AdaBoost':{},
        'Naive Bayes':{},
        'QDA':{}
        }

        # preprocess dataset, split into training and test part
        Accuracy = np.zeros((len(classifiers) + 1))
        Precision = np.zeros((len(classifiers) + 1))
        Recall = np.zeros((len(classifiers) + 1))
        F1 = np.zeros((len(classifiers) + 1))

        y_train = GA_parameters[gra]
        
        # iterate over classifiers

        for name, clf in zip(Output[gra], classifiers):
            Output[gra][name] = {
                'metrics':{
                    'Accuracy': 0,
                    'Precision':0,
                    'Recall':0,
                    'F1':0
                },
                'predict':0
            }
            try:
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)
                Output[gra][name]['predict'] = y_predict = list(clf.predict(X_test))
                Output[gra][name]['metrics']['Accuracy'] = score
                Output[gra][name]['metrics']['Precision'] = precision_score(y_test, y_predict,zero_division=0)
                Output[gra][name]['metrics']['Recall'] = recall_score(y_test, y_predict,zero_division=0)
                Output[gra][name]['metrics']['F1'] = f1_score(y_test, y_predict,zero_division=0)
            
            except:
                Output[gra]= 'Error: Only one class'
                break

    return Output

def classifiers_train(classifier,X,y):
    #Changing Work Folder
    add_path1 = "/Classification/"
    add_path2 = "/Kernel/"
    base_path = os.getcwd()
    working_path = os.getcwd()
    Classification_path = working_path + add_path1
    Kernel_path = working_path + add_path2

    # Change to Kernel directory
    os.chdir(Kernel_path)

    names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(gamma='scale'),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100),
        MLPClassifier(alpha=1,max_iter=500),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]
    
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        if name == classifier:
            clf.fit(X, y)
            pickle.dump(clf, open('model.sav', 'wb'))
            print('Model trained')
    
    # Now change to base directory

    os.chdir( base_path )

    return

def Model_Predict (projected_data,y):

    # Uploading the trained model
    
    model = pickle.load(open('Kernel/model.sav', 'rb'))
                
    score = model.score(projected_data,y)
    y_predict = list(model.predict(projected_data))
                
    Accuracy = score*100
    Precision = precision_score(y, y_predict)*100
    Recall = recall_score(y, y_predict)*100
    F1 = f1_score(y, y_predict)*100

    results = pd.DataFrame(np.hstack((Accuracy,Precision,Recall,F1)).reshape((1,-1)),columns=['Accuracy','Precision',
                                                                           'Recall','F1'])
    
    print(results)

    return
