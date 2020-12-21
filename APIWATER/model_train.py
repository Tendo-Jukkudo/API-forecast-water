import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import watermodel
import datetime
import data_function as d_f
import create_data as cd 
import os

def train_model(url_csv,save_name,type_data,his,target,asixs,id_name,f_ex,date_w,val_memory=1028,std_mean=False,gpus=False):
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
  error_date = d_f.check_datetime(url_csv,f_ex,asixs)
  nb_error = len(error_date)
  step = int(60/f_ex)
  past_history = his*24*step
  future_target = target*24*step
  io = str(int(his))+"_"+str(target)

  BATCH_SIZE = 32
  BUFFER_SIZE = 10000

  data_std,data_mean=cd.create_meanstd(url_csv,asixs)

  if(nb_error > 0):
    loss=[]
    val_loss=[]
    score_list=[]
    return loss,val_loss,data_std,data_mean,error_date,score_list

  x,y=cd.create_datatrain(url_csv,type_data,asixs,past_history,future_target,step,data_mean,data_std,std_mean)
  X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

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

  path_checkpoint = os.path.join(path_w,path_f)

  es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10)

  modelckpt_callback = tf.keras.callbacks.ModelCheckpoint(monitor="val_loss",filepath=path_checkpoint,verbose=1,save_weights_only=True,save_best_only=True)
  print("input train shape:"+str(X_train.shape))
  print("output train shape:"+str(y_train.shape))
  print("input test shape:"+str(X_test.shape))
  print("output test shape:"+str(y_test.shape))

  single_step_history = model.fit(train_data,epochs=100,validation_data=val_data,callbacks=[es_callback, modelckpt_callback])

  loss = single_step_history.history['loss']
  val_loss = single_step_history.history['val_loss']

  score_list = d_f.accuracy_score(val_data=val_data,nb_test=1000,model=model,mean=data_mean,std=data_std,type_data=type_data,mean_std=std_mean)

  return loss,val_loss,data_std,data_mean,error_date[1:],score_list
