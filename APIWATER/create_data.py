import numpy as np
import pandas as pd
import data_function as d_f

def create_meanstd(url_data,asixs):
    data_new = []
    for i in url_data:
        data_load = pd.read_csv(i)
        feature_name = list(data_load.columns)
        uni_data = data_load[feature_name[2:4]]
        uni_data = np.array(uni_data)
        data_new = d_f.array_append(np.array(data_new),uni_data)
    
    uni_data = np.array(data_new)
    if(asixs == "desc"):
        uni_data = d_f.array_reverse(uni_data)
    data_mean = uni_data[:int(len(uni_data)*0.8)].mean(axis=0)
    data_std = uni_data[:int(len(uni_data)*0.8)].std(axis=0)
    return data_std,data_mean
def create_datatrain(url_data,type_data,asixs,past_history,future_target,STEP,data_mean,data_std,mean_std=False):
    X_data=[]
    Y_data=[]
    for i in url_data:
        data_load = pd.read_csv(i)
        feature_name = list(data_load.columns)
        uni_data = data_load[feature_name[2:4]]
        uni_data = np.array(uni_data)
        if(asixs == "desc"):
            uni_data = d_f.array_reverse(uni_data)
        if(mean_std==True):
            uni_data = (uni_data-data_mean)/data_std
        x,y = d_f.multivariate_data(uni_data, uni_data[:,type_data], 0,None,int(past_history),int(future_target), step=STEP,single_step=True) 
        for e in range(0,x.shape[0]):
            X_data.append(x[e])
            Y_data.append(y[e])
    return np.array(X_data),np.array(Y_data)
