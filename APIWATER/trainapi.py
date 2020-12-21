import subprocess
from flask import Flask,request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS
from webargs.flaskparser import parser, abort
import urllib.request
import json
import time
import sys
from waitress import serve
from multiprocessing import Process, Queue
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
from threading import Thread
import threading
import time
import functools
import numpy as np
import model_train
import requests
import os

some_queue = None

url_model = 'http://hbqweblog.com/ai/model/'
url_get = 'http://hbqweblog.com/DAKWACO/stock/ai_get_data.php?'
url_csv = 'http://hbqweblog.com/ai/tdata/'
path_model = 'watermodel.py'
global nb_training
# gpus_ = bool(sys.argv[1])
# memory_gpu = int(sys.argv[2])
train_status = False

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

@APP.route('/sm/restart', methods=['GET'], endpoint='start_flaskapp')
def restart():
    try:
        path_task_manager = "model/task_manage.json"
        json_file =  open(path_task_manager)
        data = json.load(json_file)
        data['status'] = "stopped"
        data['ai_status'] = False
        write_json(data,path_task_manager)
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
    @APP.route('/sm',methods=['GET'])
    def index(): 
        return "Welcome to AI HBQsolution"
    @APP.route('/',methods=['GET'])
    def index1(): 
        return "Welcome to AI HBQsolution"

    @APP.route('/sm/ai_config',methods=['GET'])
    def ai_config():
        model_name = request.args.get('name', default = '', type = str)
        get_link = url_model+model_name+".py"
        name_model_json_file = "model/model_name.json"
        download_url(get_link,path_model, chunk_size=128)
        data = {'name':model_name}
        write_json(data,name_model_json_file)
        return "Updated"
    @APP.route('/sm/task_manage',methods=['GET'])
    def process_manage():
        try:
            path_task_manager = "model/task_manage.json"
            open_status_infor = open(path_task_manager, "r")
            infor_status = json.load(open_status_infor)
            return infor_status
        except:
            return "Error System 404"
        
    @APP.route('/sm/ai_learning',methods=['GET'])
    def returnTrain():
        global train_status
        if train_status == False:
            train = request.args.get('run', default = 0, type = int)
            id_device = request.args.get('ID', default = '', type = str)
            lin = request.args.get('lin', default = 0, type = float)
            lout = request.args.get('lout', default = 0, type = float)
            gpus_ = request.args.get('GPU', default = 0, type = int)
            memory_gpu = request.args.get('memory', default = 1028, type = int)
            gpus_ = bool(gpus_)

            named_tuple = time.localtime() 
            datetime = time.strftime("%Y-%m-%d", named_tuple)
            index = time.strftime("%Y%m%d%H%M%S", named_tuple)
            start_time = time.strftime("%Y-%m-%d, %H:%M:%S", named_tuple)

            name_model_json_file = "model/model_name.json"

            with open(name_model_json_file) as json_file: 
                data = json.load(json_file)
                model_name = data["name"]

            path = "model/"+id_device+"/"+str(index)+"/loss_data.json"
            path_weightfolder = "model/"+id_device+"/"+str(index)
            url_getdata = url_csv+id_device+"/"

            try:
                code_check = urllib.request.urlopen(url_getdata).getcode()
            except:
                code_check = 404

            if(id_device == ''):
                return jsonify({'error':"Device ID has not been inserted"})
            if(code_check == 404):
                return jsonify({'error':"Device ID does not exist"})
            if(lout==0):
                return jsonify({'error': "How much future data do you want?"})
            if(lin==0):
                return jsonify({'error': "How much past data do you want?"})

            data_link = url_getdata+"datalink.json"
            r = requests.get(data_link)
            data_links = r.json()
            list_link = data_links["source"]
            sampling = data_links["sampling"]
            asix_data = data_links["order"]
            for i in range(0,len(list_link)):
                list_link[i] = url_getdata + list_link[i]

            if(train == 1):
                train_status = True
                path_task_manager = "model/task_manage.json"
                infor_status = {'start':start_time,'ai_status':train_status,'status':"running",'ID':id_device,'error':""}
                write_json(infor_status,path_task_manager)
                loss_p,val_loss_p,std,mean,date_error,score_p=model_train.train_model(url_csv=list_link,save_name='P',
                                                                type_data=0,his=lin,target=lout,asixs=asix_data,
                                                                id_name=id_device,date_w=index,val_memory=memory_gpu,f_ex=sampling,std_mean=False,gpus=gpus_)
                loss_flow,val_loss_flow,std,mean,date_error,score_flow=model_train.train_model(url_csv=list_link,save_name='F',
                                                                        type_data=1,his=lin,target=lout,asixs=asix_data,
                                                                        id_name=id_device,date_w=index,val_memory=memory_gpu,f_ex=sampling,std_mean=True,gpus=gpus_)
                    
                if(len(date_error) > 0):
                    train_status = False
                    infor_status = {'start':start_time,'ai_status':train_status,'status':"failed",'ID':id_device,'error':date_error}
                    write_json(infor_status,path_task_manager)
                    return infor_status
                        
                training_info_json_file = "model/about_model.json"
                model_path = "model/"+id_device
                path_datetrain = "model/"+id_device+"/"+str(index)
                check_file_existed = os.path.exists(training_info_json_file)
                if(check_file_existed == False): # file chua ton tai,tao moi
                    check_id = os.path.exists(model_path)
                    check_datetrain = os.path.exists(path_datetrain)

                    if(check_id == True and check_datetrain == True):
                        data = {'infor':[{'id':id_device,'model':model_name,'lin':lin,'lout':lout,'traindate':datetime,"index":index}]}
                        write_json(data,training_info_json_file) 
                else: 
                    check_id = os.path.exists(model_path)
                    check_datetrain = os.path.exists(path_datetrain)
                    if(check_id == True and check_datetrain == True):
                        with open(training_info_json_file) as json_file: 
                            data = json.load(json_file)
                            temp = data['infor'] 
                            type_file = {'id':id_device,'model':model_name,'lin':lin,'lout':lout,'weightfolder':path_weightfolder,'traindate':datetime,"index":index}
                            temp.append(type_file) 
                        write_json(data,training_info_json_file) 

                data_out = json.dumps({'std':std,'mean':mean,'date':datetime,'model':model_name,
                                        'flow':{'loss':loss_flow,'loss_test':val_loss_flow,'score':score_flow},
                                        'pressure':{'loss':loss_p,'loss_test':val_loss_p,'score':score_p}},cls=NumpyEncoder) 
                out_file = open(path, "w")
                json.dump(data_out, out_file, indent = 6)

                train_status = False
                infor_status = {'start':start_time,'ai_status':train_status,'status':"done",'ID':id_device,'error':""}
                write_json(infor_status,path_task_manager)
            else:
                path_task_manager = "model/task_manage.json"
                open_status_infor = open(path_task_manager, "r")
                infor_status = json.load(open_status_infor)
            return infor_status
        else:
            path_task_manager = "model/task_manage.json"
            open_status_infor = open(path_task_manager, "r")
            infor_status = json.load(open_status_infor)
            return infor_status

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
