#!/usr/bin/env python

def getPatterns( path, cv, sort):

  from Gaugi import load

  def norm1( data ):
      norms = np.abs( data.sum(axis=1) )
      norms[norms==0] = 1
      #return np.expand_dims( data/norms[:,None], axis=2 )
      return data/norms[:,None]

  # Load data
  d = load(path)
  feature_names = d['features'].tolist()

  # Get the normalized rings
  data_rings = norm1(d['data'][:,1:101])
  
  # How many events?
  n = data_rings.shape[0]

  # extract all shower shapes
  data_reta   = d['data'][:, feature_names.index('reta')].reshape((n,1))
  data_rphi   = d['data'][:, feature_names.index('rphi')].reshape((n,1))
  data_eratio = d['data'][:, feature_names.index('eratio')].reshape((n,1))
  data_weta2  = d['data'][:, feature_names.index('weta2')].reshape((n,1))
  data_f1     = d['data'][:, feature_names.index('f1')].reshape((n,1))
  
  # Get the mu average 
  data_mu     = d['data'][:, feature_names.index('avgmu')].reshape((n,1))
  target = d['target']

  # This is mandatory
  splits = [(train_index, val_index) for train_index, val_index in cv.split(data_mu,target)]
  
  data_shower_shapes = np.concatenate( (data_reta,data_rphi,data_eratio,data_weta2,data_f1), axis=1)

    # split for this sort
  x_train = [ data_rings[splits[sort][0]],   data_shower_shapes [ splits[sort][0] ] ]
  x_val   = [ data_rings[splits[sort][1]],   data_shower_shapes [ splits[sort][1] ] ]
  y_train = target [ splits[sort][0] ]
  y_val   = target [ splits[sort][1] ]

  return x_train, x_val, y_train, y_val, splits


def getPileup( path ):
  from Gaugi import load
  return load(path)['data'][:,0]


def getJobConfigId( path ):
  from Gaugi import load
  return dict(load(path))['id']


import numpy as np
import os


#
# reference configuration
#

from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
stop = EarlyStopping(monitor='val_sp', mode='max', verbose=1, patience=25, restore_best_weights=True)

import datetime, os
from tensorflow.keras.callbacks import TensorBoard
logdir = os.path.join('.', 'logs/%s' %(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
tensorboard = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)


#from saphyra.metrics import sp_metric, pd_metric, fa_metric
from AutoencoderReconsturctionJob import AutoencoderReconsturctionJob


job = AutoencoderReconsturctionJob (  
                                loss              = 'binary_crossentropy',
                                metrics           = 'accuracy',
                                batch_size        = 500,
                                callbacks         = [stop, tensorboard],
                                epochs            = 50,
                                class_weight      = True,
                                sorts             = 1,
                                inits             = 1,
                                )

# Run it!
job.run()
