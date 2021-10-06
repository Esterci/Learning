import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import pickle as pk


#########################################################
# ------------------------------------------------------ #
# ----------------------- LOADING ---------------------- #
# ------------------------------------------------------ #
##########################################################
# Firstly the model loads the background and signal data, 
# then it removes the attributes first string line, which 
# are the column names, in order to avoid NaN values in 
# the array.

print('==== Commencing Initiation ====\n')

### Background
b_name='/home/thiago/Documents/Data_Sets/LPC-anomaly-detection/Input_Background_1.csv'
background = np.genfromtxt(b_name, delimiter=',')
background = background[1:,:]
print(".Background Loaded..." )
print(".Background shape: {}".format(background.shape))

### Signal
s_name='/home/thiago/Documents/Data_Sets/LPC-anomaly-detection/Input_Signal_1.csv'
signal = np.genfromtxt(s_name, delimiter=',')
signal = signal[1:,:]
print(".Signal Loaded...")
print(".Signal shape: {}\n".format(signal.shape))

##########################################################
# ------------------------------------------------------ #
# --------------------- INITIATION --------------------- #
# ------------------------------------------------------ #
##########################################################

# Number of events
total = 500000

# Percentage of background samples on the testing phase
background_percent = 0.99

# Percentage of samples on the training phase
test_size = 0.3

# Number of iterations

n_it = 33


for it in range(n_it):

    print('\n          ==== Initiation Complete ====\n')
    print('=*='*17 )
    print('\n      ==== Commencing Pre-processing ====\n')

    # Percentage of background samples to divide the data-set
    dat_set_percent = total/len(background)

    # Reducing background samples
    _,reduced_background = train_test_split(background, test_size=dat_set_percent)

    # Deviding train and test background

    train_data, background_test = train_test_split(reduced_background, test_size=test_size)

    # Iserting the correct number of signal in streaming

    n_signal_samples = int(len(background_test)*(1-background_percent))

    _,background_test = train_test_split(background_test, test_size=background_percent)

    _,signal_test = train_test_split(signal, test_size=n_signal_samples/len(signal))

    # Concatenating Signal and the Background sub-sets

    test_data = np.vstack((background_test,signal_test))

    # Normalize Data

    print('.Normalizing Data')

    test_data = normalize(test_data,
                            norm='max',
                            axis=0
                        )

    train_data = normalize(train_data,
                            norm='max',
                            axis=0
                        )

    # Creates test data frame

    attributes = np.array(["px1",
                            "py1",
                            "pz1",
                            "E1",
                            "eta1",
                            "phi1",
                            "pt1",
                            "px2",
                            "py2",
                            "pz2",
                            "E2",
                            "eta2",
                            "phi2",
                            "pt2",
                            "Delta_R",
                            "M12",
                            "MET",
                            "S",
                            "C",
                            "HT",
                            "A"]
                        )

    test_df = pd.DataFrame(test_data,columns = attributes)

    # Creating Labels
    print('.Creating Labels')

    test_labels =np.ones((len(test_data)))
    test_labels[:len(background_test)] = 0

    print('\n      ==== Pre-processing Complete ====\n')
    print(".Train data shape: {}".format(train_data.shape))
    print(".Test data shape: {}".format(test_data.shape))
    print(".Test Background shape: {}".format(background_test.shape))
    print(".Test Signal shape: {}".format(signal_test.shape))

    print('=*='*17 )

    Output = {'train_df'  : train_data,
              'test_df'     : test_df,
              'test_labels' : test_labels}

    struct_name = ('data-divisions/data__total__' + str(total) +
                   '__background_percent__' + str(background_percent) +
                   '__test_size__' + str(test_size) +
                   '__n_it__' + str(it) + '__.pkl')
    
    with open(struct_name, 'wb') as f:
        pk.dump(Output, f)

