import os
import glob
import numpy as np

# define the paths into the container
data_path  = 'Data_Base/*'

# Defining hyper-parameters range

min_batch_size = 200

max_batch_size = 500

min_hidden_dim = 8

max_hidden_dim = 21

# Parameters in study

#batch_size_list = list(np.linspace(max_batch_size,min_batch_size,num=3,dtype=int))
batch_size_list = [200]
#encoding_dim_list = list(np.linspace(min_hidden_dim,max_hidden_dim,num=3,dtype=int))
encoding_dim_list = [8]
#lambda_disco_list = list(np.linspace(600,0,num=3,dtype=int))
lambda_disco_list = [600]
#act_func_list_1 = ['relu',
                   #'sigmoid',
                   #'softmax',
                   #'softplus',
                   #'softsign',
                   #'tanh',
                   #'selu',
                   #'elu',
                   #'exponential'
                   #]
act_func_list_1 = ['tanh']
#act_func_list_2 = act_func_list_1
act_func_list_2 = ['sigmoid']
#act_func_list_3 = act_func_list_1
act_func_list_3 = ['sigmoid']
attributes = np.linspace(0,20,num=21,dtype=int)


# create a list of config files
file_list  = glob.glob(data_path)

for file in file_list:
    for batch_size in batch_size_list:
        for encoding_dim in encoding_dim_list:
            for lambda_disco in lambda_disco_list:
                for act_1 in act_func_list_1:
                    for act_2 in act_func_list_2:
                        for act_3 in act_func_list_3:
                            for dcorr in attributes:
                                m_command = """python3 autoencoder_Dijets.py -b {BACH} \\
                                -e {EDIM} \\
                                -l {LAMBDA} \\
                                -a1 {A1} \\
                                -a2 {A2} \\
                                -a3 {A3} \\
                                -f {FILE}\\
                                -dcorr {CORR}""".format(BACH=batch_size, 
                                                        EDIM=encoding_dim,
                                                        LAMBDA=lambda_disco,
                                                        A1=act_1,
                                                        A2=act_2,
                                                        A3=act_3,
                                                        FILE=file,
                                                        CORR=dcorr)

                                print(m_command)
                                # execute the tuning
                                os.system(m_command)
