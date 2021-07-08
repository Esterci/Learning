import numpy as np
from progress.bar import Bar

print('====== Welcome to the Cross-Validation program ======')

output_id = input('which input you want to format?')

full_data = np.genfromtxt('Input/Output_' + output_id + '.csv',
                                delimiter=',')
    
L, W = full_data.shape

data = full_data[:,2:-1]
info = full_data[:,0:2]