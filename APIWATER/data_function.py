import numpy as np
import pandas as pd

def multivariate_data(dataset, target, start_index, end_index, history_size,target_size, step, single_step=False):
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

def add_predict(dataset,nb_predict):
  a = [0,0]
  data_new = []
  for i in range(0,dataset.shape[0]):
    data_new.append(dataset[i])
  for i in range(0,nb_predict):
    data_new.append(a)
  return data_new

def datatime_json(Now_time,data_predict,f_ex):
    data_json = []
    for i in range(0,len(data_predict)):
      Now_time = np.datetime64(Now_time) + np.timedelta64(f_ex, 'm')
      data_json.append([str(Now_time),round(data_predict[i][0],2)])
    return data_json

def check_datestep(data_time,f_ex):
  data_residual = []
  step=60/f_ex
  sum_date = 24*step
  ct = 0
  time_1 = 0
  time_2 = 0
  for i in range(0,len(data_time)):
    ct = ct + 1
    ct_1 = 0
    ct_2 = 0
    for e in data_time[i-1][0]:
      ct_1 = ct_1 + 1
      if (e == '-'):
        ct_2=ct_2+1
        if (ct_2 == 2):
          time_1 = data_time[i-1][0][ct_1-8:ct_1+2]
          # print(*["time_after:",time_1])
    ct_3 = 0
    ct_4 = 0
    for e in data_time[i][0]:
      ct_3 = ct_3 + 1
      if (e == '-'):
        ct_4=ct_4+1
        if (ct_4 == 2):
          time_2 = data_time[i][0][ct_3-8:ct_3+2]
          # print(*["time_last:",time_2])
    if (time_2 != time_1):
      if (ct < sum_date or ct > sum_date):
        data_residual.append([time_1,ct])
      ct = 0
  return data_residual

def filter_time(dataset):
  data_new = []
  for i in range(0,dataset[:,:1].shape[0]):
    count = 0
    for e in dataset[:,:1][i][0]:
      count = count + 1 
      if(str(e) == ':'):
        if (str(dataset[:,:1][i][0][count:count+2]) == '00' or str(dataset[:,:1][i][0][count:count+2]) == '15' or str(dataset[:,:1][i][0][count:count+2]) == '30' or str(dataset[:,:1][i][0][count:count+2]) == '45'):
          data_new.append(dataset[i])
  return data_new

def array_reverse(dataset):
    data_new = []
    for i in range(dataset.shape[0]-1,-1,-1):
      data_new.append(dataset[i])
    return np.array(data_new)

def array_append(dataset,data_append):
  data_new = []
  for i in range(0,dataset.shape[0]):
      data_new.append(dataset[i])
  for i in range(0,data_append.shape[0]):
      data_new.append(data_append[i])
  return data_new

def check_datetime(url_data,f_ex):
    e_date=[]
    for i in url_data:
        data_load = pd.read_csv(i)
        feature_name = list(data_load.columns)
        datetime = data_load[feature_name[1:2]]
        datetime = np.array(datetime)
        error_date = check_datestep(datetime,f_ex)
        error_date = error_date[1:]
        for e in range(0,len(error_date)):
            e_date.append(error_date[e])
    return e_date