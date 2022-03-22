import glob
import pickle as pk
import sys
from SODA_numba import SelfOrganisedDirectionAwareDataPartitioning as SODA
import numpy as np


### defining granularities to study

granularities = [2,4,6,8]

### creates a dictionary for the divisions

# define the paths into the container

data_path  = '/home/thiago/Repositories/Learning/Python/Anomaly_Detection/distribution-analysis/data/divisions/*'

# create a list of config files

file_list  = glob.glob(data_path)

# creating data divisions dictionary

divisions_dict = {}

# getting number of divisions

n_elements = len(file_list)

print('Reading reductions...')

for i,file_name in enumerate(file_list):

    with open(file_name, 'rb') as f:
        seed = pk.load(f)

    divisions_dict[i] = seed

    # updating progress bar

    percent = (i+1)/n_elements * 100/2

    info = '{:.2f}% - {:d} of {:d}'.format(percent*2,(i+1),n_elements)

    formated_bar = '-'*int(percent) + ' '*int(50-percent)

    if i < (n_elements):
        sys.stdout.write("\r")

    sys.stdout.write('[%s] %s' % (formated_bar,info))
    sys.stdout.flush()



### creates a dictionary for the divisions

# define the paths into the container


print('\n\nInitiating SODA\n')


data = divisions_dict[0]['train_df']

SODA_input = {
    'StaticData' : data[0:100],
    'GridSize' : 1,
    'DistanceType' : 'euclidean'
}

SODA(SODA_input)


print('\n\nList of granularities to study: {}\n'.format(granularities))

for gra in granularities:

    for div in divisions_dict:

        data = divisions_dict[div]['train_df']

        SODA_input = {
            'StaticData' : data[:],
            'GridSize' : gra,
            'DistanceType' : 'euclidean'
        }

        print('\n\nAppling SODA with granularity {} on division {}...\n'.format(gra,div))


        output = SODA(SODA_input)


        struct_name = 'outputs/granularity__{}__iteration__{}__.pkl'.format(gra,div)
                            

        with open(struct_name, 'wb') as f:
            pk.dump(output, f)

