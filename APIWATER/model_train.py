import tensorflow as tf
from sklearn.model_selection import train_test_split
from contextlib import redirect_stdout
import numpy as np
import pandas as pd
import watermodel
import datetime
import data_function as d_f
import create_data as cd 
import logging
import time
import os

LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (20, 0.01),
    (40, 0.001),
]
class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logging.info(
            "The average loss for epoch {} is {:7.3f}.".format(epoch, logs["val_loss"])
        )

class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        logging.info("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))

def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr

def train_model(url_csv,save_name,type_data,row_infor,his,target,asixs,id_name,f_ex,date_w,batch_size,val_memory=1028,std_mean=False,gpus=False):

  print("Request training Device:"+id_name)
  logging.info("Request training Device:"+id_name)
  if(gpus==True):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
      try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=val_memory)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

  step = int(60/f_ex)
  past_history = his*24*step
  future_target = target*24*step
  io = str(int(his))+"_"+str(target)

  BATCH_SIZE = batch_size
  BUFFER_SIZE = 10000

  
  print("Data processing ...")
  logging.info("Data processing ...")
  x,y,data_mean,data_std,date_time=cd.create_datatrain(url_csv,type_data,row_infor,asixs,past_history,future_target,step,std_mean)
  print("Data processed")
  logging.info("Data processed")
  date_time = list(map(str,date_time))
  print("Data Checking ...")
  logging.info("Data Checking ...")
  error_date = d_f.check_datestep(date_time,f_ex)
  nb_error = len(error_date)


  if(nb_error > 0):
    loss=[]
    val_loss=[]
    score_list=[]
    print("Data Error")
    logging.error("Data Error No: " +nb_error)
    print(error_date)
    return loss,val_loss,data_std,data_mean,error_date,score_list
    
  print("Good Data")
  logging.info("Good Data")
  print("input shape:"+str(x.shape))
  print("output shape:"+str(y.shape))

  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  print("input train shape:"+str(X_train.shape))
  print("output train shape:"+str(y_train.shape))
  print("input test shape:"+str(X_test.shape))
  print("output test shape:"+str(y_test.shape))
  print("Start Training...")
  logging.info("Start Training...")
  train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
  train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

  val_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
  val_data = val_data.batch(BATCH_SIZE)
  model = watermodel.model_main(input_shape=X_train.shape[-2:]) 

  parent_dir = "model/"
  directory = id_name+"/"
  path = os.path.join(parent_dir,directory)
  check = os.path.exists(path)
  if(check == False):
      os.mkdir(path)
  directory_w = date_w+"/"
  path_w = os.path.join(path,directory_w)
  check_w = os.path.exists(path_w)
  if(check_w == False):
      os.mkdir(path_w)
        
  path_f = "W"+save_name+".h5"

  log_dir = "logs/fit/"+id_name+"/"+date_w
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

  path_checkpoint = os.path.join(path_w,path_f)
  es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10)
  modelckpt_callback = tf.keras.callbacks.ModelCheckpoint(monitor="val_loss",filepath=path_checkpoint,verbose=1,save_weights_only=True,save_best_only=True)

  with open('status/model.log', 'w') as f:
    with redirect_stdout(f):
        model.summary()

  single_step_history = model.fit(train_data,epochs=100,
                                  validation_data=val_data,
                                  callbacks=[es_callback, modelckpt_callback,LossAndErrorPrintingCallback(),CustomLearningRateScheduler(lr_schedule),tensorboard_callback])

  print("Successfully Training Process")
  logging.info("Successfully Training Process")
  loss = single_step_history.history['loss']
  val_loss = single_step_history.history['val_loss']

  print("PROCESS END")
  logging.info("PROCESS END")
  
  return loss,val_loss,data_std,data_mean,error_date[1:]
