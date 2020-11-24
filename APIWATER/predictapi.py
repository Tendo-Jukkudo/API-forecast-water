import subprocess
from flask import Flask,request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS
from webargs.flaskparser import parser, abort
import json
import time
import sys
from waitress import serve
from multiprocessing import Process, Queue
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
import functools
import numpy as np
import model_predict
import requests
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

some_queue = None

frequency_ex = 15
url_model = 'http://hbqweblog.com/ai/model/watermodel.zip'
url_get = 'http://hbqweblog.com/DAKWACO/stock/ai_get_data.php?'
url_csv = 'http://hbqweblog.com/ai/tdata/'
path_model = 'watermodel.py'

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
def exists(path):
    r = requests.head(path)
    return r.status_code == requests.codes.ok
def write_json(data, filename='data.json'): 
    with open(filename,'w') as f: 
        json.dump(data, f, indent=4) 

APP = Flask(__name__)
API = Api(APP)
CORS(APP)

@APP.route('/sm/restart/', methods=['GET'], endpoint='start_flaskapp')
def restart():
    try:
        some_queue.put("something")
        print("Restarted successfully")
        return("Quit")
    except:
        print("Failed in restart")
        return "Failed"

def start_flaskapp(queue):
    global some_queue
    some_queue = queue
    API.add_resource(FractionsResource, "/")
    serve(APP, host='0.0.0.0', port=8080, threads=2)

def long_function():
    with ProcessPool(5) as pool:
        data = [0, 1, 2, 3, 4]
        future = pool.map(functools.partial(add_const, const=1), data, timeout=5)
        iterator = future.result()
        result=[]
        while True:
            try:
                result.append(next(iterator))
            except StopIteration:
                break
            except TimeoutError as error:
                print("function took longer than %d seconds" % error.args[1])
    return(result)

def long_function_without_pool():
    data = [0, 1, 2, 3, 4]
    result = list(map(functools.partial(add_const, const=1), data))
    return(result)

def add_const(number, const=0):
    time.sleep(5)
    return number+const

class FractionsResource(Resource):
    @APP.route('/sm/ai_run',methods=['GET'])
    def returnPredict():
        predict = request.args.get('run', default = 0, type = int)
        id_device = request.args.get('ID', default = '', type = str)
        date = request.args.get('date', default = '2020-01-01', type = str)
        date_w = request.args.get('datew', default = '2020-01-01', type = str)
        lin = request.args.get('lin', default = 0, type = float)
        lout = request.args.get('lout', default = 0, type = float)
        url_getdata = (url_get+'&dev_id='+id_device+'&date='+date)

        io = str(int(lin))+"-"+str(lout)

        path_info = "model/W"+id_device+"/"+str(int(lin))+"-"+str(lout)+"/"+date_w+"/loss_data.json"
        check_info = os.path.exists(path_info)
        check_model = os.path.exists(path_model)

        if(check_model==False):
            download_url(url_model,path_model)

        if(check_info==False):
            return "Device ID "+id_device+" not be trainning"
        else:
            in_file = open(path_info, "r")
            data_out = json.load(in_file)
            data_out = json.loads(data_out)

            mean_value = data_out['mean']
            std_value = data_out['std']

            path_w_f = "model/W"+id_device+"/"+str(int(lin))+"-"+str(lout)+"/"+date_w+"/WF.h5"
            path_w_p = "model/W"+id_device+"/"+str(int(lin))+"-"+str(lout)+"/"+date_w+"/WP.h5"
            parent_dir = "data/"
            directory = id_device+"/"
            path = os.path.join(parent_dir,directory)
            check = os.path.exists(path)
            if(check == False):
                os.mkdir(path)
            directory_c = io+"/"
            path_c = os.path.join(path,directory_c)
            check_c = os.path.exists(path_c)
            if(check_c == False):
                os.mkdir(path_c)

            path_f = date+".json"

            path_save = os.path.join(path_c,path_f)

            if(predict == 1):
                forecast_flow,error_date = model_predict.run_prediction(type_feature=1,
                                                                his=lin,target=lout,
                                                                path_weight=path_w_f,url_get=url_getdata,
                                                                means=mean_value,f_ex=frequency_ex,stds=std_value,mean_std=True)
                forecast_p,error_date = model_predict.run_prediction(type_feature=0,
                                                                his=lin,target=lout,
                                                                path_weight=path_w_p,url_get=url_getdata,
                                                                means=mean_value,f_ex=frequency_ex,stds=std_value,mean_std=False)

                data_out = json.dumps({'status':date,'error':error_date,'flow':forecast_flow,'Pressure':forecast_p},cls=NumpyEncoder) 
                out_file = open(path_save, "w")
                json.dump(data_out, out_file, indent = 6)    
            else:
                in_file = open(path_save, "r")
                data_out = json.load(in_file) 
            return jsonify({'output':data_out})

if __name__ == "__main__":

    q = Queue()
    p = Process(target=start_flaskapp, args=(q,))
    p.start()
    while True: #wathing queue, if there is no call than sleep, otherwise break
        if q.empty():
            time.sleep(1)
        else:
            break
    p.terminate() #terminate flaskapp and then restart the app on subprocess
    args = [sys.executable] + [sys.argv[0]]
    subprocess.call(args)