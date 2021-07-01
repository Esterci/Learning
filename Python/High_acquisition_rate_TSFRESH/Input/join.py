import numpy as np
from progress.bar import Bar

print('====== Welcome to the joining program ======')

list_id = [3,13]

concat_output = np.genfromtxt('Output_' + str(list_id[0]) + '.csv',delimiter=',')  

with Bar('Concatenating...') as bar:
    for i in range(1,len(list_id)):
        aux = np.genfromtxt('Output_' + str(list_id[i]) + '.csv',delimiter=',')
        concat_output = np.concatenate((concat_output, aux), axis=0)
        bar.next(100/len(range(1,len(list_id))))

new_id = '3_13'

np.savetxt('Output_' + new_id + '.csv',concat_output,delimiter=',')
