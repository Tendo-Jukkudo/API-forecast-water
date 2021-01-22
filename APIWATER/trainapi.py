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
import logging
import numpy as np
import model_train
import requests
import zipfile 
import ftplib 
import os

some_queue = None

url_model = 'http://hbqweblog.com/ai/model/'
url_get = 'http://hbqweblog.com/DAKWACO/stock/ai_get_data.php?'
url_csv = 'http://sovigaz.hbqweblog.com/ai/tdata/'
path_model = 'watermodel.py'
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
def zipit(folders, zip_filename):
    zip_file = zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED)

    for folder in folders:
        for dirpath, dirnames, filenames in os.walk(folder):
            for filename in filenames:
                zip_file.write(
                    os.path.join(dirpath, filename),
                    os.path.relpath(os.path.join(dirpath, filename), os.path.join(folders[0], '../..')))

    zip_file.close()

def send_ftp(folders_path,zipname,FTP_HOST,FTP_USER,FTP_PASS):
    # //Nen folder folder_path
    zipit(folders_path, zipname)
    # //dang nhap tai khoan FTP
    ftp = ftplib.FTP(FTP_HOST, FTP_USER, FTP_PASS)
    # force UTF-8 encoding
    ftp.encoding = "utf-8"
    # //send file nen
    filename = "some_file.txt"
    with open(zipname, "rb") as file:
        # use FTP's STOR command to upload the file
        ftp.storbinary(f"STOR {zipname}", file)
    # //return ketqua
    os.remove(zipname)



APP = Flask(__name__)
API = Api(APP)
CORS(APP)

@APP.route('/sm/restart', methods=['GET'], endpoint='start_flaskapp')
def restart():
    try:
        logging.info("Training process stopped")
        path_task_manager = "model/task_manage.json"
        json_file = open(path_task_manager)
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

    @APP.route('/sm/ai_console',methods=['GET'])
    def ai_console():
        train_status = open("status/train_status.log","r")
        return "<pre>"+train_status.read()+"</pre>"

    @APP.route('/sm/ai_model',methods=['GET'])
    def ai_model():
        model_status = open("status/model.log","r")
        return "<pre>"+model_status.read()+"</pre>"


    @APP.route('/sm/ai_learning',methods=['GET'])
    def returnTrain():
        global train_status
        if train_status == False:
            train = request.args.get('run', default = 0, type = int)
            id_device = request.args.get('ID', default = '', type = str)
            lin = request.args.get('lin', default = 0, type = float)
            lout = request.args.get('lout', default = 0, type = float)
            gpus_ = request.args.get('GPU', default = 0, type = int)
            batch_size = request.args.get('batchsize', default = 32, type = int)
            memory_gpu = request.args.get('memory', default = 1028, type = int)
            gpus_ = bool(gpus_)

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

            if(train == 1):
                logging.FileHandler(filename="status/train_status.log", mode='w')
                logging.basicConfig(filename="status/train_status.log", level=logging.INFO)
                train_status = True

                named_tuple = time.localtime() 
                datetime = time.strftime("%Y-%m-%d", named_tuple)
                index = time.strftime("%Y%m%d%H%M%S", named_tuple)
                start_time = time.strftime("%Y-%m-%d, %H:%M:%S", named_tuple)

                path_task_manager = "model/task_manage.json"
                infor_status = {'start':start_time,'ai_status':train_status,'status':"running",'ID':id_device,'error':""}
                write_json(infor_status,path_task_manager)

                url_getdata = url_csv+id_device+"/"

                data_link = url_getdata+"datalink.json"
                r = requests.get(data_link)
                data_links = r.json()
                list_link = data_links["source"]
                sampling = data_links["sampling"]
                asix_data = data_links["order"]
                row_datetime_name = data_links["rowname"][0]
                row_pressure_name = data_links["rowname"][1]
                row_flow_name = data_links["rowname"][2]
                row_data = [row_datetime_name,row_pressure_name,row_flow_name]

                for i in range(0,len(list_link)):
                    list_link[i] = url_getdata + list_link[i]

                loss_p,val_loss_p,std,mean,date_error=model_train.train_model(url_csv=list_link,save_name='P',
                                                                type_data=0,row_infor=row_data,his=lin,target=lout,asixs=asix_data,
                                                                id_name=id_device,date_w=index,batch_size=batch_size,val_memory=memory_gpu,f_ex=sampling,std_mean=False,gpus=gpus_)
                loss_flow,val_loss_flow,std,mean,date_error=model_train.train_model(url_csv=list_link,save_name='F',
                                                                        type_data=1,row_infor=row_data,his=lin,target=lout,asixs=asix_data,
                                                                        id_name=id_device,date_w=index,batch_size=batch_size,val_memory=memory_gpu,f_ex=sampling,std_mean=True,gpus=gpus_)
                if(len(date_error) > 0):
                    train_status = False
                    infor_status = {'start':start_time,'ai_status':train_status,'status':"failed",'ID':id_device,'error':date_error}
                    write_json(infor_status,path_task_manager)
                    return infor_status

                name_model_json_file = "model/model_name.json"

                with open(name_model_json_file) as json_file: 
                    data = json.load(json_file)
                    model_name = data["name"]

                path_weightfolder = "model/"+id_device+"/"+str(index)        
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
                                        'flow':{'loss':loss_flow,'loss_test':val_loss_flow},
                                        'pressure':{'loss':loss_p,'loss_test':val_loss_p}},cls=NumpyEncoder) 
                path = "model/"+id_device+"/"+str(index)+"/loss_data.json"                        
                out_file = open(path, "w")
                json.dump(data_out, out_file, indent = 6)

                train_status = False
                infor_status = {'start':start_time,'ai_status':train_status,'status':"done",'ID':id_device,'error':""}
                write_json(infor_status,path_task_manager)
                logging.info("Proceed to send the file")
                try:
                    send_ftp(folders_path=["model","logs"],zipname="mnt/result_model.zip",
                            FTP_HOST="hbqweblog.com",
                            FTP_USER="ftpuser",
                            FTP_PASS="Thehoang091184")
                    logging.info("The folder has been sent")
                except:
                    logging.error("Folder has not been sent")
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
    date_strftime_format = "%d-%b-%y %H:%M:%S"
    message_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(filename="status/train_status.log",
                        format=message_format,datefmt=date_strftime_format,
                        level=logging.INFO)
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
