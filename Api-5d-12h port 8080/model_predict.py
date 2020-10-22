import tensorflow as tf
from keras.layers import Conv1D,Dense,MaxPooling1D,Bidirectional,LSTM,Input,GlobalAveragePooling1D
from keras.layers import Dropout,LeakyReLU,BatchNormalization,Concatenate,add
from keras.models import Model
from keras import optimizers
import requests
import numpy as np
import pandas as pd

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

def DBL(filters,kernel_size,n,connect_layer):
  x = Conv1D(filters,kernel_size,padding='same',activation=LeakyReLU(alpha=0.1))(connect_layer)
  x = BatchNormalization(axis=1)(x)
  x = Dropout(0.2)(x)
  for i in range(1,n):
    x = Conv1D(filters,kernel_size,padding='same',activation=LeakyReLU(alpha=0.1))(x)
    x = BatchNormalization(axis=1)(x)
    x = Dropout(0.2)(x)
  return x

def res_block1D(filters,alpha,beta,n,connect_layer):
  x = MaxPooling1D()(connect_layer)
  x = DBL(filters,kernel_size=2,n=1,connect_layer=x)
  for i in range(0,n):
    previous_block_activation = x
    x = DBL(alpha,kernel_size=1,n=1,connect_layer=x)
    x = DBL(beta,kernel_size=2,n=1,connect_layer=x)
    x = add([x,previous_block_activation])
  return x

def model_predict(inputs_shape):
  input_layer = Input(shape=inputs_shape)
  x = DBL(filters=32,kernel_size=1,n=1,connect_layer=input_layer)
  x = res_block1D(connect_layer = x,filters=64,alpha=32,beta=64,n=1)
  x = res_block1D(connect_layer = x,filters=128,alpha=64,beta=128,n=2)
  x = res_block1D(connect_layer = x,filters=256,alpha=128,beta=256,n=8)
  x = GlobalAveragePooling1D()(x)
  x1 = LSTM(256,activation = 'tanh',dropout=0.3)(input_layer)
  concat_layer = Concatenate(axis=1)([x1,x])
  x2 = Dense(64,activation=LeakyReLU(alpha=0.1))(concat_layer)
  x2 = Dense(1,activation = 'linear')(x2)
  model = Model(inputs=input_layer,outputs=x2)
  return model

def add_predict(dataset,nb_predict):
  a = [0,0]
  data_new = []
  for i in range(0,dataset.shape[0]):
    data_new.append(dataset[i])
  for i in range(0,nb_predict):
    data_new.append(a)
  return data_new

def futures_predict(input_data,type_data,path_weights,past_history,future_target,STEP,mean_std=False):
  data_mean = [1.779176,16.71776386]
  data_std = [0.35215948, 9.0908301]
  predict_data = []
  true_data = []
  if(mean_std==True):
    uni_data = (input_data - data_mean)/data_std
    uni_data = add_predict(uni_data,future_target)
    uni_data = np.array(uni_data)
    x_data, y_data = multivariate_data(uni_data, uni_data[:,type_data],0, None, past_history,future_target, STEP,single_step=True)

    X_data = x_data[x_data.shape[0]-future_target:x_data.shape[0]]
    Y_data = y_data[x_data.shape[0]-future_target:x_data.shape[0]] 

    test_data = tf.data.Dataset.from_tensor_slices((X_data,Y_data))
    test_data = test_data.batch(1)

    model = model_predict(x_data.shape[-2:])
    model.load_weights(path_weights)
    for x, y in test_data.take(future_target):
      data_p = model.predict(x)[0]*data_std[type_data] + data_mean[type_data]
      predict_data.append(data_p) 
  else:
    uni_data = input_data
    uni_data = add_predict(uni_data,future_target)
    uni_data = np.array(uni_data)
    x_data, y_data = multivariate_data(uni_data, uni_data[:,type_data],0, None, past_history,future_target, STEP,single_step=True)

    X_data = x_data[x_data.shape[0]-future_target:x_data.shape[0]]
    Y_data = y_data[x_data.shape[0]-future_target:x_data.shape[0]] 

    test_data = tf.data.Dataset.from_tensor_slices((X_data,Y_data))
    test_data = test_data.batch(1)

    model = model_predict(x_data.shape[-2:])
    model.load_weights(path_weights)
    for x, y in test_data.take(future_target):
      data_p = model.predict(x)[0]
      predict_data.append(data_p) 

  return predict_data

def datatime_json(Now_time,data_predict):
    data_json = []
    for i in range(0,len(data_predict)):
      Now_time = np.datetime64(Now_time) + np.timedelta64(15, 'm')
      data_json.append([str(Now_time),round(data_predict[i][0],2)])
    return data_json

def check_datestep(data_time):
  data_residual = []
  ct = 0
  time_1 = 0
  time_2 = 0
  for i in range(0,len(data_time)):
    ct = ct + 1
    ct_1 = 0
    ct_2 = 0
    for e in data_time[i-1]:
      ct_1 = ct_1 + 1
      if (e == '-'):
        ct_2=ct_2+1
        if (ct_2 == 2):
          time_1 = data_time[i-1][ct_1-8:ct_1+2]
          # print(*["time_after:",time_1])
    ct_3 = 0
    ct_4 = 0
    for e in data_time[i]:
      ct_3 = ct_3 + 1
      if (e == '-'):
        ct_4=ct_4+1
        if (ct_4 == 2):
          time_2 = data_time[i][ct_3-8:ct_3+2]
          # print(*["time_last:",time_2])
    if (time_2 != time_1):
      if (ct < 96 or ct > 96):
        data_residual.append([time_1,ct,i])
      ct = 0
  return data_residual

def run_prediction(type_feature,nb_past,nb_future,path_model,url_get,mean_std=False):
  r = requests.get(url_get)
  df = pd.DataFrame(r.json())
  feature_name = list(df.columns)
  uni_data = df[feature_name[1:3]]
  uni_data = np.array(uni_data)

  datetime = df[feature_name[0:1]]
  datetime = np.array(datetime)
  nb_error = len(check_datestep(datetime))
  if(nb_error == 0):
    data_predict = futures_predict(input_data=uni_data,type_data=type_feature,path_weights=path_model,past_history=nb_past,future_target=nb_future,STEP=4,mean_std=mean_std)
    Now_time = datetime[datetime.shape[0]-1][0]
    data_json = datatime_json(Now_time,data_predict)
    output = np.array(data_json)
    error_date = check_datestep(datetime)
  else:
    error_date = check_datestep(datetime)
    output = "Error data input"
  return output,error_date