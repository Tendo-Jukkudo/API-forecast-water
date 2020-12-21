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

    @APP.route('/sm',methods=['GET'])
    def index(): 
        return "Welcome to AI HBQsolution"

    @APP.route('/',methods=['GET'])
    def index1(): 
        return "Welcome to AI HBQsolution"

    @APP.route('/sm/ai_view',methods=['GET'])
    def returnView():
        try:
            date = request.args.get('date', default = '', type = str)
            id_device = request.args.get('ID', default = '', type = str)

            if(id_device == '' and len(date) > 0):
                return 'Please enter the ID'

            data_info_json_file = "data/about_data.json"
            with open(data_info_json_file) as json_file: 
                data = json.load(json_file)
                temp = data['infor']
            if(date == '' and id_device == ''):
                return data
            if(len(date) > 0):
                data_out = [data_ for data_ in temp if data_['predictdate'] == date][0]
                file_json_date = data_out['predictdate']
                path_data = "data/"+id_device+"/"+file_json_date+".json"
                with open(path_data) as json_file: 
                    data_predict = json.load(json_file)
                return jsonify({'output':data_predict})
            if(len(id_device) > 0):
                data_out = [data_ for data_ in temp if data_['id'] == id_device]
                return jsonify({'infor':data_out})  
        except:
            return "System Error 404"
    
    @APP.route('/sm/ai_analyze',methods=['GET'])  
    def return_ws():
        try:
            id_device = request.args.get('ID', default = '', type = str)
            index = request.args.get('index', default = '', type = str)

            training_info_json_file = "model/about_model.json"

            with open(training_info_json_file) as json_file: 
                data = json.load(json_file)
                temp = data['infor']
            if(id_device == ''):
                return data
            elif(len(index) > 0):
                datas = [data_ for data_ in temp if data_['index'] == index][0]
                weightfolder = datas['weightfolder']+"/"+"loss_data.json"
                with open(weightfolder) as json_file: 
                    loss_infor = json.load(json_file)
                return loss_infor
            else:
                datas = [data_ for data_ in temp if data_['id'] == id_device]
                return jsonify({'infor':datas})  
        except:
            return "System Error 404"

    @APP.route('/sm/ai_run',methods=['GET'])
    def returnPredict():
        predict = request.args.get('run', default = 0, type = int)
        id_device = request.args.get('ID', default = '', type = str)
        date = request.args.get('date', default = '', type = str)
        index = request.args.get('index', default = '', type = str)

        training_info_json_file = "model/about_model.json"
        path_id_device = 'data/'+id_device
        check_id_device = os.path.exists(path_id_device)

        if(check_id_device == False):
            return "Device ID "+id_device+" not be trainning"
        else:
            url_getdata = url_get+'&dev_id='+id_device+'&date='+date
            url_getlink = url_csv+id_device+"/"
            data_link = url_getlink+"datalink.json"
            r = requests.get(data_link)
            data_links = r.json()
            sampling = data_links["sampling"]
                
            with open(training_info_json_file) as json_file: 
                data = json.load(json_file)
                temp = data['infor']

            datas = [data_ for data_ in temp if data_['index'] == index][0]
            lin = datas['lin']
            lout = datas['lout']
            path_info = datas['weightfolder']+"/"+"loss_data.json"

            in_file = open(path_info, "r") # open file loss_data
            loss_data = json.load(in_file) 
            loss_data = json.loads(loss_data)

            mean_value = loss_data['mean']
            std_value = loss_data['std']

            path_w_f = datas['weightfolder']+"/WF.h5"
            path_w_p = datas['weightfolder']+"/WP.h5"
            parent_dir = "data/"
            directory = id_device+"/"
            path = os.path.join(parent_dir,directory)
            check = os.path.exists(path)
            if(check == False):
                os.mkdir(path)

            path_f = date+".json"

            path_save = os.path.join(path,path_f)

            name_model_json_file = "model/model_name.json"

            with open(name_model_json_file) as json_file: 
                data = json.load(json_file)
                model_name = data["name"]

            if(predict == 1):
                data_info_json_file = "data/about_data.json"
                path_id = "model/"+id_device
                path_date_predict = "data/"+id_device+"/"+date+".json"
                check_info_json_file = os.path.exists(data_info_json_file)
                status = "running"
                if(check_info_json_file== False):
                    data = {'infor':[{'id':id_device,'predictdate':date,'model':model_name,'lin':lin,'lout':lout,'weightfolder':datas['weightfolder'],"status":status}]}
                    write_json(data,data_info_json_file) 
                else:
                    check_id = os.path.exists(path_id)
                    check_date_predict = os.path.exists(path_date_predict)
                    if(check_id == False or check_date_predict == False):
                        with open(data_info_json_file) as json_file: 
                            data = json.load(json_file)
                            temp = data['infor'] 
                            type_file = {'id':id_device,'predictdate':date,'model':model_name,'lin':lin,'lout':lout,'weightfolder':datas['weightfolder'],"status":status}
                            temp.append(type_file) 
                        write_json(data,data_info_json_file)
                    if(check_id == True or check_date_predict == True):
                        with open(data_info_json_file) as json_file: 
                            data = json.load(json_file)
                            temp = data['infor'] 
                            get_status = [data_ for data_ in temp if data_['predictdate'] == date][0]
                            get_status['status'] = status
                            get_status['weightfolder'] = datas['weightfolder']
                            get_status['lin'] = lin
                            get_status['lout'] = lout
                        write_json(data,data_info_json_file)

                forecast_flow,error_date = model_predict.run_prediction(type_feature=1,
                                                                    his=lin,target=lout,
                                                                    path_weight=path_w_f,url_get=url_getdata,
                                                                    means=mean_value,f_ex=sampling,stds=std_value,mean_std=True)
                forecast_p,error_date = model_predict.run_prediction(type_feature=0,
                                                                    his=lin,target=lout,
                                                                    path_weight=path_w_p,url_get=url_getdata,
                                                                    means=mean_value,f_ex=sampling,stds=std_value,mean_std=False)
                status = "done"
                with open(data_info_json_file) as json_file: 
                    data = json.load(json_file)
                    temp = data['infor'] 
                    get_status = [data_ for data_ in temp if data_['predictdate'] == date][0]
                    get_status['status'] = status
                write_json(data,data_info_json_file)

                data_out = json.dumps({'status':date,'error':error_date,'flow':forecast_flow,'Pressure':forecast_p},cls=NumpyEncoder) 
                out_file = open(path_save, "w")
                json.dump(data_out, out_file, indent = 6)    
        return "OK"

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