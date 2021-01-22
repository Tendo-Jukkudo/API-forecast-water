import tensorflow as tf
import watermodel
import requests
import numpy as np
import pandas as pd
import os
import data_function as d_f

def futures_predict(input_data,type_data,path_weights,past_history,future_target,STEP,mean,std,mean_std=False):
  data_mean = mean
  data_std = std
  predict_data = []
  true_data = []
  uni_data = d_f.add_predict(input_data,future_target)
  uni_data = np.array(uni_data)
  x_data, y_data = d_f.multivariate_data(uni_data, uni_data[:,type_data],0, None, past_history,future_target, STEP,single_step=True)

  X_data = x_data[x_data.shape[0]-future_target:x_data.shape[0]]
  Y_data = y_data[x_data.shape[0]-future_target:x_data.shape[0]] 

  test_data = tf.data.Dataset.from_tensor_slices((X_data,Y_data))
  test_data = test_data.batch(1)

  model = watermodel.model_main(input_shape=X_data.shape[-2:])
  model.load_weights(path_weights)
  for x, y in test_data.take(future_target):
    if(mean_std == True):
      data_p = model.predict(x)[0]*data_std[type_data] + data_mean[type_data]
    else:
      data_p = model.predict(x)[0]
    predict_data.append(data_p) 
  return predict_data

def run_prediction(type_feature,his,target,path_weight,url_get,f_ex,means,stds,mean_std=False):

  nb_past = his*24*4
  nb_future = target*24*4
  step = 60/f_ex

  data_mean = means
  data_std = stds

  r = requests.get(url_get)
  df = pd.DataFrame(r.json())
  feature_name = list(df.columns)
  uni_data = df[feature_name[1:3]]
  uni_data = np.array(uni_data)
  if(mean_std==True):
    uni_data = (uni_data - data_mean)/data_std

  datetime = list(map(str,df[0]))
  nb_error = len(d_f.check_datestep(datetime,f_ex))
  if(nb_error == 0):
    data_predict = futures_predict(input_data=uni_data,type_data=type_feature,
                                    path_weights=path_weight,
                                    past_history=int(nb_past),future_target=int(nb_future),
                                    STEP=int(step),mean=data_mean,std=data_std,mean_std=mean_std)
    Now_time = datetime[len(datetime)-1]
    data_json = d_f.datatime_json(Now_time,data_predict,f_ex)
    output = np.array(data_json)
    error_date = d_f.check_datestep(datetime,f_ex)
  else:
    error_date = d_f.check_datestep(datetime,f_ex)
    output = []
  return output,error_date[1:]