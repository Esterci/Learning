import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle as pk
import sys
from scipy import stats

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

print('\n          ==== Initiation Complete ====\n')
print('=*='*17 )

##########################################################
# ------------------------------------------------------ #
# --------------------- INITIATION --------------------- #
# ------------------------------------------------------ #
##########################################################

# Number of events
total = 100000

# Percentage of background samples on the testing phase
background_percent = 0.99

# Percentage of samples on the training phase
test_size = 0.3

# Number of iterations

n_it = 33

# Defining type of division

prob = True


attributes = ["px1","py1","pz1","E1","eta1",
                "phi1","pt1","px2","py2","pz2",
                "E2","eta2","phi2","pt2",
                "Delta_R","M12","MET","S","C",
                "HT","A"]


if prob == False:

    # Percentage of background samples to divide the data-set

    dat_set_percent = total/len(background)

    # getting number of divisions

    n_elements = n_it


    # iniciating progress bar

    print('\n\nCreating iterations...')

    percent = (0)/n_elements * 100/2

    info = '{:.2f}% - {:d} of {:d}'.format(percent*2,(0),n_elements)

    formated_bar = '-'*int(percent) + ' '*int(50-percent)

    sys.stdout.write("\r")

    sys.stdout.write('[%s] %s' % (formated_bar,info))
    sys.stdout.flush()


    for it in range(n_it):


        # Reducing background samples

        _,reduced_background = train_test_split(background, test_size=dat_set_percent)


        # Deviding train and test background

        background_train, background_test = train_test_split(reduced_background, test_size=test_size)


        # Iserting the correct number of signal in training

        n_signal_samples = int(len(background_train)*(1-background_percent))

        _,background_train = train_test_split(background_train, test_size=background_percent)

        _,signal_train= train_test_split(signal, test_size=n_signal_samples/len(signal))


        # Concatenating Signal and the Background sub-sets

        train_data = np.vstack((background_train,signal_train))


        # Iserting the correct number of signal in streaming

        n_signal_samples = int(len(background_test)*(1-background_percent))

        _,background_test = train_test_split(background_test, test_size=background_percent)

        _,signal_test = train_test_split(signal, test_size=n_signal_samples/len(signal))


        # Concatenating Signal and the Background sub-sets

        test_data = np.vstack((background_test,signal_test))


        # Normalize Data

        scaler = MinMaxScaler()

        test_data = scaler.fit_transform(test_data)

        train_data = scaler.fit_transfor(train_data)


        # Creates test data frame

        test_df = pd.DataFrame(test_data,columns = attributes)

        train_df = pd.DataFrame(train_data,columns = attributes)


        # Creating Labels

        test_labels = np.ones((len(test_data)))

        test_labels[:len(background_test)] = 0

        train_labels =np.ones((len(train_data)))

        train_labels[:len(background_train)] = 0


        Output = {'train_df'  : train_df,
                'test_df'     : test_df,
                'test_labels' : test_labels,
                'train_labels' : train_labels
                }


        struct_name = ('data/divisions/data__total__' + str(total) +
                    '__background_percent__' + str(background_percent) +
                    '__test_size__' + str(test_size) +
                    '__n_it__' + str(it) + '__.pkl')
                    
        
        with open(struct_name, 'wb') as f:
            pk.dump(Output, f)

        # updating progress bar

        percent = (it+1)/n_elements * 100/2

        info = '{:.2f}% - {:d} of {:d}'.format(percent*2,(it+1),n_elements)

        formated_bar = '-'*int(percent) + ' '*int(50-percent)

        if it < (n_elements):
            sys.stdout.write("\r")

        sys.stdout.write('[%s] %s' % (formated_bar,info))
        sys.stdout.flush()
    
    sys.stdout.write("\r")


else:

    ### computing most probable events

    try:

        with open('data/seeds/z_score__.pkl', 'rb') as f:

            z_score = pk.load(f)     

        bg_z_score_list = z_score['bg_z_score']

        sg_z_score_list = z_score['sg_z_score']

        print('-> z_score values loaded')

    
    except:


        # creating z_score lists

        bg_z_score_list = []

        sg_z_score_list = []

        background_len = len(background)+len(signal)

        signal_len = len(background)+len(signal)


        # getting number of divisions

        n_elements = len(background[0])


        # iniciating progress bar

        print('\n\nCalculating z_score values...')

        percent = (0)/n_elements * 100/2

        info = '{:.2f}% - {:d} of {:d}'.format(percent*2,(0),n_elements)

        formated_bar = '-'*int(percent) + ' '*int(50-percent)

        sys.stdout.write("\r")

        sys.stdout.write('[%s] %s' % (formated_bar,info))

        sys.stdout.flush()


        for i in range(len(background[0])):

            # getting attributes values

            att_background = background[:,i]

            att_signal = signal[:,i]


            # unite background and signal for z-score

            # calculation

            all_data = np.hstack((att_background,att_signal))

            mean = np.mean(all_data)

            std = np.std(all_data)


            # calculation zscore

            bg_z_score = (((att_background - mean)/std)**2)**0.5
            
            sg_z_score = (((att_signal - mean)/std)**2)**0.5


            # creating z_score lists

            bg_z_score_list.append(bg_z_score)

            sg_z_score_list.append(sg_z_score)


            # updating progress bar

            percent = (i+1)/n_elements * 100/2

            info = '{:.2f}% - {:d} of {:d}'.format(percent*2,(i+1),n_elements)

            formated_bar = '-'*int(percent) + ' '*int(50-percent)

            if i < (n_elements):
                sys.stdout.write("\r")

            sys.stdout.write('[%s] %s' % (formated_bar,info))

            sys.stdout.flush()
        
        sys.stdout.write("\r")


        z_score = {
            'bg_z_score' : bg_z_score_list,
            'sg_z_score' : sg_z_score_list
        }

        with open('data/seeds/z_score__.pkl', 'wb') as f:

            pk.dump(z_score, f)

    ### organizing background and signal using z_score

    # Background

    sum_bg_z_score = np.sum(bg_z_score_list,axis=0)

    bg_args = np.argsort(sum_bg_z_score)

    sum_bg_z_score = sum_bg_z_score[bg_args]


    # Signal

    sum_sg_z_score = np.sum(sg_z_score_list,axis=0)

    sg_args = np.argsort(sum_sg_z_score)

    sum_sg_z_score = sum_sg_z_score[sg_args]


    # Seting proportoin variables

    n_background = int(total * background_percent)

    n_signal = int(total - n_background)


    # Reducing samples

    reduced_background = background[bg_args[:n_background]]

    reduced_signal = signal[sg_args[:n_signal]]


    # Concatenating Signal and the Background sub-sets

    data = np.vstack((reduced_background,reduced_signal))


    # Normalize Data

    scaler = MinMaxScaler()

    data = scaler.fit_transform(data)


    # Creates test data frame

    df = pd.DataFrame(data,columns = attributes)


    # Creating Labels

    labels =np.ones((len(data)))

    labels[:len(reduced_background)] = 0
    

    Output = {'df'  : df,
            'labels' : labels
            }


    struct_name = ('data/prob-reductions/reduced_data__total__' + str(total) +
                '__background_percent__' + str(background_percent) +
                '__.pkl')
                
    
    with open(struct_name, 'wb') as f:
        pk.dump(Output, f)

