#!/usr/bin/env python


def getJobConfigId( path ):
  from Gaugi import load
  return dict(load(path))['id']


import argparse
import sys,os
import numpy as np

parser = argparse.ArgumentParser(description = '', add_help = False)
parser = argparse.ArgumentParser()


parser.add_argument('-c','--configFile', action='store',
        dest='configFile', required = True,
            help = "The job config file that will be used to configure the job (sort and init).")

parser.add_argument('-v','--volume', action='store',
        dest='volume', required = False, default = None,
            help = "The volume output.")


if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

args = parser.parse_args()

##########################################################
# ------------------------------------------------------ #
# --------------------- INITIATION --------------------- #
# ------------------------------------------------------ #
##########################################################

# Defining hyper-parameters range

min_batch_size = 200

max_batch_size = 500

min_lambda = 0

max_lambda = 600

batch_size_list = list(np.linspace(min_batch_size,max_batch_size,num=1,dtype=int))

lambda_disco_list = list(np.linspace(max_lambda,min_lambda,num=1,dtype=int))

job_id = getJobConfigId( args.configFile )

for l_disco in lambda_disco_list:
  for batch in batch_size_list:

    try:

      outputFile = args.volume+'/tunedDiscr.jobID_%s'%str(job_id).zfill(4) if args.volume else 'test.jobId_%s'%str(job_id).zfill(4)

      from tensorflow.keras.callbacks import EarlyStopping
      import tensorflow as tf
      stop = EarlyStopping(monitor='val_sp', mode='max', verbose=1, patience=25, restore_best_weights=True)

      import datetime, os
      from tensorflow.keras.callbacks import TensorBoard
      logdir = os.path.join('.', 'logs/%s' %(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
      tensorboard = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

      #from saphyra.applications import BinaryClassificationJob
      from AutoencoderReconsturctionJob import AutoencoderReconsturctionJob


      job = AutoencoderReconsturctionJob( job               = args.configFile,
                                          epochs        = 150,
                                          batch_size    = batch,
                                          lambda_disco  = l_disco,
                                          metrics           = ['accuracy'],
                                          callbacks         = [stop, tensorboard],
                                          outputFile        = outputFile )

      # Run it!
      job.run()


      # necessary to work on orchestra
      from saphyra import lock_as_completed_job
      lock_as_completed_job(args.volume if args.volume else '.')

      sys.exit(0)

    except  Exception as e:
      print(e)

      # necessary to work on orchestra
      from saphyra import lock_as_failed_job
      lock_as_failed_job(args.volume if args.volume else '.')

      sys.exit(1)



