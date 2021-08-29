
__all__ = ["create_jobs"]

from Gaugi import retrieve_kw, mkdir_p
from Gaugi.messenger import Logger
from Gaugi.messenger.macros import *
from Job_v1 import Job_v1


# A simple solution need to refine the documentation
from itertools import product
def create_iter(fun, n_items_per_job, items_lim):
  return ([fun(i, n_items_per_job)
           if (i+n_items_per_job) <= items_lim 
           else fun(i, items_lim % n_items_per_job) 
           for i in range(0, items_lim, n_items_per_job)])


# default model (ringer vanilla)
# Remove the keras dependence and get keras from tensorflow 2.0
import tensorflow as tf
default_model = tf.keras.Sequential()
default_model.add(tf.keras.layers.Dense(5, input_shape=(100,), activation='tanh', kernel_initializer='random_uniform', bias_initializer='random_uniform'))
default_model.add(tf.keras.layers.Dense(1, activation='linear', kernel_initializer='random_uniform', bias_initializer='random_uniform'))
default_model.add(tf.keras.layers.Activation('tanh'))
 

class create_configuration_jobs:
  '''
  Documentation (TODO)
  '''

  def __init__( self, **kw):

    self.outputFolder        = retrieve_kw( kw, 'outputFolder' ,       'jobConfig'           )
    self.sortBounds          = retrieve_kw( kw, 'sortBounds'   ,             5               )
    self.nInits              = retrieve_kw( kw, 'nInits'       ,             10              )
    self.nSortsPerJob        = retrieve_kw( kw, 'nSortsPerJob' ,             1               )
    self.nInitsPerJob        = retrieve_kw( kw, 'nInitsPerJob' ,             10              ) 
    self.nModelsPerJob       = retrieve_kw( kw, 'nModelsPerJob',             1               ) 
    self.models              = retrieve_kw( kw, 'models'       ,   [default_model]           )
    self.model_tags          = retrieve_kw( kw, 'model_tags'   ,   ['mlp_100_5_1']           )

  def time_stamp(self):
    from datetime import datetime
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H.%M.%S")
    return timestampStr

  def create_jobs( self, **kw):     

    time_stamp = self.time_stamp()    
    # creating the job mechanism file first
    mkdir_p(self.outputFolder)

    if type(self.models) is not list:
      self.models = [self.models]
    
    modelJobsWindowList = create_iter(lambda i, sorts: list(range(i, i+sorts)), 
                                      self.nModelsPerJob,
                                      len(self.models))
    sortJobsWindowList  = create_iter(lambda i, sorts: list(range(i, i+sorts)), 
                                      self.nSortsPerJob,
                                      self.sortBounds)
    initJobsWindowList  = create_iter(lambda i, sorts: list(range(i, i+sorts)), 
                                      self.nInitsPerJob, 
                                      self.nInits)

    nJobs = 0 
    for (model_idx_list, sort_list, init_list) in product(modelJobsWindowList,
                                                          sortJobsWindowList, 
                                                          initJobsWindowList):

      job = Job_v1()
      # to be user by the database table
      job.setId( nJobs )
      job.setSorts(sort_list)
      job.setInits(init_list)
      job.setModels([self.models[idx] for idx in model_idx_list],  model_idx_list )
      # save config file
      model_str = 'ml%i.mu%i' %(model_idx_list[0], model_idx_list[-1])
      sort_str  = 'sl%i.su%i' %(sort_list[0], sort_list[-1])
      init_str  = 'il%i.iu%i' %(init_list[0], init_list[-1])
      job.save( self.outputFolder+'/' + ('job_config.ID_%s.%s_%s_%s.%s') %
              ( str(nJobs).zfill(4), model_str, sort_str, init_str, time_stamp) )
      nJobs+=1
