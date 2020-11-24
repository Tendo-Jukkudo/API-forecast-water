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
import model_train
import requests
import os

some_queue = None

frequency_ex = 15
url_model = 'http://hbqweblog.com/ai/model/watermodel.zip'
url_get = 'http://hbqweblog.com/DAKWACO/stock/ai_get_data.php?'
url_csv = 'http://hbqweblog.com/ai/tdata/'
path_model = 'watermodel.py'

gpus_ = bool(sys.argv[1])
memory_gpu = int(sys.argv[2])

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
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

APP = Flask(__name__)
API = Api(APP)
CORS(APP)

@APP.route('/sm/restart/', methods=['GET'], endpoint='start_flaskapp')
def restart():
    try:
        some_queue.put("something")
        print("\nRestarted successfully")
        return("Quit")
    except:
        print("Failed in restart")
        return "Failed"

def start_flaskapp(queue):
    global some_queue
    some_queue = queue
    API.add_resource(FractionsResource, "/")
    serve(APP, host='0.0.0.0', port=5000, threads=2)

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
    @APP.route('/sm/ai_learning',methods=['GET'])
    def returnTrain():
        train = request.args.get('run', default = 0, type = int)
        id_device = request.args.get('ID', default = '', type = str)
        lin = request.args.get('lin', default = 0, type = float)
        lout = request.args.get('lout', default = 0, type = float)

        named_tuple = time.localtime() 
        datetime = time.strftime("%Y-%m-%d", named_tuple)
        print(datetime)
        path = "model/W"+id_device+"/"+str(int(lin))+"-"+str(lout)+"/"+str(datetime)+"/loss_data.json"
        url_getdata = [url_csv+id_device+".csv"]

        #download_url(url_model,path_model)

        if(id_device == ''):
            return "Device ID has not been inserted"

        if(train == 1):
            loss_p,val_loss_p,std,mean,date_error=model_train.train_model(url_csv=url_getdata,save_name='P',
                                                        type_data=0,his=lin,target=lout,
                                                        id_name=id_device,val_memory=memory_gpu,f_ex=frequency_ex,std_mean=False,gpus=gpus_)
            loss_flow,val_loss_flow,std,mean,date_error=model_train.train_model(url_csv=url_getdata,save_name='F',
                                                                type_data=1,his=lin,target=lout,
                                                                id_name=id_device,val_memory=memory_gpu,f_ex=frequency_ex,std_mean=True,gpus=gpus_)
            
            if(len(date_error) > 0):
                return jsonify({'output':date_error})
            
            data_out = json.dumps({'std':std,'mean':mean,'date':datetime,'model':path_model,
                                    'flow':{'loss':loss_flow,'loss_test':val_loss_flow},
                                    'pressure':{'loss':loss_p,'loss_test':val_loss_p}},cls=NumpyEncoder) 
            out_file = open(path, "w")
            json.dump(data_out, out_file, indent = 6)

        else:
            check_f = os.path.exists(path)
            if(check_f == False):
                return "Found not"
            else:
                with open(path) as f: 
                    data_out= json.load(f)
        return jsonify({'evaluate':data_out})

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
    args = [sys.executable] + [sys.argv[0]] + [sys.argv[1]] + [sys.argv[2]]
    subprocess.call(args)