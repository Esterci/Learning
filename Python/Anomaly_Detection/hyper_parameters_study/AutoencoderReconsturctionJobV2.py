

__all__ = ['BinaryClassificationJob', 'lock_as_completed_job', 'lock_as_failed_job']

from Gaugi.messenger import Logger
from Gaugi.messenger.macros import *
from Gaugi import StatusCode, checkForUnusedVars, retrieve_kw

from tensorflow.keras.models import clone_model
from tensorflow.keras import backend as K
from sklearn.utils.class_weight import compute_class_weight

from datetime import datetime
from copy import deepcopy
import numpy as np
import time
import os

def lock_as_completed_job(output):
  with open(output+'/.complete','w') as f:
    f.write('complete')

def lock_as_failed_job(output):
  with open(output+'/.failed','w') as f:
    f.write('failed')





class BinaryClassificationJob( Logger ):

  def __init__(self , pattern_generator, crossval, **kw ):

    Logger.__init__(self)

    self.__pattern_generator = pattern_generator
    self.crossval = crossval


    self.optimizer       = retrieve_kw( kw, 'optimizer'      , 'adam'                )
    self.loss            = retrieve_kw( kw, 'loss'           , 'binary_crossentropy' )
    self.epochs          = retrieve_kw( kw, 'epochs'         , 1000                  )
    self.batch_size      = retrieve_kw( kw, 'batch_size'     , 1024                  )
    self.callbacks       = retrieve_kw( kw, 'callbacks'      , []                    )
    self.metrics         = retrieve_kw( kw, 'metrics'        , []                    )
    self.sorts           = retrieve_kw( kw, 'sorts'          , range(1)              )
    self.inits           = retrieve_kw( kw, 'inits'          , 1                     )
    job_auto_config      = retrieve_kw( kw, 'job'            , None                  )
    self.__verbose       = retrieve_kw( kw, 'verbose'        , True                  )
    self.__class_weight  = retrieve_kw( kw, 'class_weight'   , False                 )
    self.__save_history  = retrieve_kw( kw, 'save_history'   , True                  )
    self.decorators      = retrieve_kw( kw, 'decorators'     , []                    )
    self.plots           = retrieve_kw( kw, 'plots'          , []                    )
    self.__model_generator=retrieve_kw( kw, 'model_generator', None                  )

    # read the job configuration from file
    if job_auto_config:
      if type(job_auto_config) is str:
        MSG_INFO( self, 'Reading job configuration from: %s', job_auto_config )
        from saphyra.core.readers import JobReader
        job = JobReader().load( job_auto_config )
      else:
        job = job_auto_config
      # retrive sort/init lists from file
      self.sorts = job.getSorts()
      self.inits = job.getInits()
      self.__models, self.__id_models = job.getModels()
      self.__jobId = job.id()


    # get model and tag from model file or lists
    models = retrieve_kw( kw, 'models', None )
    if models:
      self.__models = models
      self.__id_models = [id for id in range(len(models))]
      self.__jobId = 0



    self.__outputfile = retrieve_kw( kw, 'outputFile' , None           )

    if self.__outputfile:
      from saphyra.core.readers.versions import TunedData_v1
      self.__tunedData = TunedData_v1()

    checkForUnusedVars(kw)


    from saphyra import Context
    self.__context = Context()


    self.__trained_models = []



  #
  # Sorts setter and getter
  #
  @property
  def sorts(self):
    return self.__sorts

  @sorts.setter
  def sorts( self, s):
    if type(s) is int:
      self.__sorts = range(s)
    else:
      self.__sorts = s


  #
  # Init setter and getter
  #
  @property
  def inits(self):
    return self.__inits

  @inits.setter
  def inits( self, s):
    if type(s) is int:
      self.__inits = range(s)
    else:
      self.__inits = s



  #
  # run job
  #
  def run( self ):



    for isort, sort in enumerate( self.sorts ):

      # get the current kfold and train, val sets
      x_train, x_val = self.pattern_g( self.__pattern_generator, self.crossval, sort )

      # check if there are fewer events than the batch_size
      _, n_evt_per_class = np.unique(y_train, return_counts=True)
      batch_size = (self.batch_size if np.min(n_evt_per_class) > self.batch_size
                     else np.min(n_evt_per_class))

      MSG_INFO( self, "Using %d as batch size.", batch_size)

      for imodel, model in enumerate( self.__models ):

        for iinit, init in enumerate(self.inits):

          # force the context is empty for each training
          self.__context.clear()
          self.__context.setHandler( "jobId"    , self.__jobId         )
          self.__context.setHandler( "valData"  , (x_val, x_val)       )
          self.__context.setHandler( "trnData"  , (x_train, x_train)   )


          print(model)
          # get the model "ptr" for this sort, init and model index
          if self.__model_generator:
            MSG_INFO( self, "Apply model generator..." )
            model_for_this_init = self.__model_generator( sort )
          else: 
            model_for_this_init = clone_model(model) # get only the model


          try:
            model_for_this_init.compile( self.optimizer,
                      loss = self.loss,
                      # protection for functions or classes with internal variables
                      # this copy avoid the current training effect the next one.
                      metrics = deepcopy(self.metrics),
                      #metrics = self.metrics,
                      )
            model_for_this_init.summary()
          except RuntimeError as e:
            MSG_FATAL( self, "Compilation model error: %s" , e)


          MSG_INFO( self, "Training model id (%d) using sort (%d) and init (%d)", self.__id_models[imodel], sort, init )
          MSG_INFO( self, "Train Samples      :  (%d, %d)", len(y_train[y_train==1]), len(y_train[y_train!=1]))
          MSG_INFO( self, "Validation Samples :  (%d, %d)", len(y_val[y_val==1]),len(y_val[y_val!=1]))

          self.__context.setHandler( "model"   , model_for_this_init     )
          self.__context.setHandler( "sort"    , sort                    )
          self.__context.setHandler( "init"    , init                    )
          self.__context.setHandler( "imodel"  , self.__id_models[imodel])



          callbacks = deepcopy(self.callbacks)
          for callback in callbacks:
            if hasattr(callback, 'set_validation_data'):
              callback.set_validation_data( (x_val,y_val) )


          start = datetime.now()

          if self.__class_weight:
            classes = np.unique(y_train)
            weights = compute_class_weight('balanced',classes,y_train)
            class_weights = {cl : weights[idx] for idx, cl in enumerate(classes)}
          else:
            class_weights = None

         
          # Hacn: used by orchestra to set this job as local test
          if os.getenv('LOCAL_TEST'): 
            MSG_INFO(self, "The LOCAL_TEST environ was detected." )
            MSG_INFO(self, "This is a short local test, lets skip the fitting for now. ")
            return StatusCode.SUCCESS


          # Training
          history = model_for_this_init.fit(x_train, y_train,
                              epochs          = self.epochs,
                              batch_size      = batch_size,
                              verbose         = self.__verbose,
                              validation_data = (x_val,y_val),
                              # copy protection to avoid the interruption or interference
                              # in the next training (e.g: early stop)
                              callbacks       = callbacks,
                              class_weight    = class_weights,
                              shuffle         = True).history

          end = datetime.now()

          self.__context.setHandler("time" , end-start)


          if not self.__save_history:
            # overwrite to slim version. This is used to reduce the output size
            history = {}


          self.__context.setHandler( "history", history )


          for tool in self.decorators:
            #MSG_INFO( self, "Executing the pos processor %s", tool.name() )
            tool.decorate( history, self.__context )

          
          for plot in self.plots:
            plot( self.__context )


          # add the tuned parameters to the output file
          if self.__outputfile:
            self.__tunedData.attach_ctx( self.__context )


          # Clear everything for the next init
          K.clear_session()


          self.__trained_models.append( (model_for_this_init, history) )

      # You must clean everythin before reopen the dataset
      self.__context.clear()
      # Clear the keras once again just to be sure
      K.clear_session()


    # End of training
    try:
      # prepare to save the tuned data
      if self.__outputfile:
        self.__tunedData.save( self.__outputfile )
    except Exception as e:
      MSG_FATAL( self, "Its not possible to save the tuned data: %s" , e )


    return StatusCode.SUCCESS




  def pattern_g( self, generator, crossval, sort ):
    # If the index is not set, you muat run the cross validation Kfold to get the index
    # this generator must be implemented by the user
    return generator(crossval, sort)



  def getAllModels(self):
    return self.__trained_models


