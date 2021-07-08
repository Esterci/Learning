import unified_bib as uf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from tsfresh.feature_extraction import extract_features, ComprehensiveFCParameters
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.feature_extraction import feature_calculators
from tsfresh.feature_extraction import EfficientFCParameters
import pickle
import pandas as pd
import tsfresh
from psutil import cpu_percent
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
import matplotlib.pyplot as plt

output_id = '100'
full_data = np.genfromtxt('Input/Output_' + output_id + '.csv',
                                delimiter=',')

for it in range(iterações):

    # Spliting test and train sets

    target = full_data[:,len(full_data[0])-1]
    ID = full_data[:,0]

    good_id = np.unique(ID[target==0]).tolist()
    bad_id = np.unique(ID[target==1]).tolist()

    good_train, good_test = train_test_split(good_id, test_size=0.3)
    bad_train, bad_test = train_test_split(bad_id,test_size=0.3)

    train_index =[]
    test_index =[]

    for i in range(len(full_data)):
        if (ID[i] in good_train) or (ID[i] in bad_train):
            train_index.append(i)
            
        else:
            test_index.append(i)

    train_set = full_data[train_index,:]
    test_set = full_data[test_index,:]

    _,train_set_target = np.unique(train_set[:,0],return_index=True)
    _,test_set_target = np.unique(test_set[:,0],return_index=True)

    train_set_target = train_set[train_set_target.tolist(),8]
    test_set_target = test_set[test_set_target.tolist(),8]

    # TSFRESH on the training set

    SelectedFeatures = uf.tsfresh_chucksize(train_set,output_id)

    # PCA on training set

    ReducedFeatures = uf.PCA_calc(SelectedFeatures,3,'Calc')