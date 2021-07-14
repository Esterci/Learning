import unified_bib as uf
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle
import matplotlib.pyplot as plt

###### Input Variables
n_folds = 5
output_id = '107'

# Filtering NaN in TSFRESH Features

print('Checking NaN Features in TSFRESH')

try:
    with open('Kernel/valid_features_dict__' + output_id + '.pkl', 'rb') as f:
        kind_to_fc_parameters = pickle.load(f)
except:
    uf.tsfresh_NaN_filter(output_id)

# Preparing informations for the cross validations subsets

full_data = np.genfromtxt('Input/Output_' + output_id + '.csv',
                                delimiter=',')

full_data = np.delete(full_data,len(full_data[0])-1,1)

print(full_data.shape)

target = full_data[:,-1]
ID = full_data[:,0]

unique_id, id_idx= np.unique(ID,return_index=True)
rskf_target = target[id_idx]

rskf = StratifiedKFold(n_splits=n_folds)

for it,(kf_train_idx, kf_test_idx) in enumerate(rskf.split(unique_id,rskf_target)):

    print('#################### Iteration {} ####################'.format(it))
    
    # Divinding train and test data

    id_train = [unique_id[i] for i in kf_train_idx]
    id_test = [unique_id[i] for i in kf_test_idx]

    train_index =[]
    test_index =[]

    for i in range(len(ID)):
        if ID[i] in id_train:
            train_index.append(i)
            
        elif ID[i] in id_test:
            test_index.append(i)

    train_set = full_data[train_index,:]
    test_set = full_data[test_index,:]

    _,train_index_first_occurance = np.unique(train_set[:,0],return_index=True)
    _,test_index_first_occurance = np.unique(test_set[:,0],return_index=True)

    train_set_target = train_set[train_index_first_occurance.tolist(),-1]
    test_set_target = test_set[test_index_first_occurance.tolist(),-1]

    # Extracting and selecting features

    SelectedFeatures = uf.tsfresh_chucksize(train_set,output_id)

    with open('Classification/relevance_table__' + output_id + '__' + str(it) + '__.pkl', 'wb') as f:
            pickle.dump(SelectedFeatures['RelevanceTable'], f)

    # Reducing dimensionality using PCA

    ReducedFeatures = uf.PCA_calc(SelectedFeatures,4,'Calc')

    # Data partitioning by SODA

    SODA_parameters = uf.SODA(ReducedFeatures,1,10,1) 

    # Executing a Grouping algorithm with hard voting

    GA_parameters = uf.GroupingAlgorithm(SODA_parameters)

    # Testing phase

    extracted_features = uf.dynamic_tsfresh(test_set, 'test')

    projected_features = uf.PCA_projection(extracted_features,4)

    Classifiers_result = uf.non_parametric_classification(ReducedFeatures['ReducedFeatures'],projected_features,
                        GA_parameters,test_set_target)

    with open('Classification/Classifiers_result__' + output_id + '__' + str(it) + '__.pkl', 'wb') as f:
            pickle.dump(Classifiers_result, f)

    training_data = ReducedFeatures['ReducedFeatures']
    test_data = projected_features

    fig = plt.figure(figsize=[14,10])
    ax = fig.add_subplot(111, projection='3d')

    # Treino

    ax.scatter(training_data[train_set_target== 0,0], 
            training_data[train_set_target== 0,1], 
            training_data[train_set_target== 0,2], 
            c = 'b',label='adequate tool on training')

    ax.scatter(training_data[train_set_target == 1,0], 
            training_data[train_set_target == 1,1], 
            training_data[train_set_target == 1,2], 
            c = 'r',label='worn tool on training')

    # Teste
            
    ax.scatter(test_data[test_set_target== 0,0], 
            test_data[test_set_target== 0,1], 
            test_data[test_set_target== 0,2], 
            c = 'g',label='adequate tool on test')

    ax.scatter(test_data[test_set_target == 1,0], 
            test_data[test_set_target == 1,1], 
            test_data[test_set_target == 1,2], 
            c = 'purple',label='worn tool on test')

    plt.ylabel('PC2',fontsize = 20,labelpad=18)
    plt.xlabel('PC1',fontsize = 20, labelpad=18)
    ax.set_zlabel('PC3', fontsize = 20, labelpad=12)
    plt.tick_params(axis='x', labelsize=16)
    plt.tick_params(axis='y', labelsize=16)
    plt.tick_params(axis='z', labelsize=16)

    ax.grid()
    plt.legend()

    fig.savefig('PCA_visualization__' + output_id + '__' + str(it) + '__.png', bbox_inches='tight')
