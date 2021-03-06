from flask import Flask, request, jsonify
import model_predict
import json
import numpy as np
import time 

path_model_flow = 'model/model_flow.h5'
path_model_p = 'model/model_p.h5'
url_get = 'http://hbqweblog.com/DAKWACO/stock/ai_get_data.php?dev_id=F98420'
past_history = 480
future_target = 48
            
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

app = Flask(__name__)
@app.route('/sm/ai_run',methods=['GET'])
def returnPredict():
    run = request.args.get('run', default = 0, type = int)
    date = request.args.get('date', default = '2020-01-01', type = str)
    url_getdata = (url_get+'&date='+date)
    if(run == 1):
        in_file = open("data_predict.json", "r")
        data = json.load(in_file) 
        data = json.loads(data)
        flow = data['flow']
        Pressure = data['Pressure']
        forecast_flow,error_date = model_predict.run_prediction(type_feature=1,
                                                                nb_past=past_history,nb_future=future_target,
                                                                path_model=path_model_flow,url_get=url_getdata,mean_std=True)
        forecast_p,error_date = model_predict.run_prediction(type_feature=0,
                                                             nb_past=past_history,nb_future=future_target,
                                                             path_model=path_model_p,url_get=url_getdata,mean_std=True)
        if forecast_flow[len(forecast_flow)-1][0] != flow[len(flow)-1][0]:
            for i in forecast_flow:
                flow.append(i)
            for i in forecast_p:
                Pressure.append(i)
        named_tuple = time.localtime() 
        time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
        data_out = json.dumps({'flow':flow,'Pressure':Pressure,'status':time_string,'error':error_date},cls=NumpyEncoder) 
        out_file = open("data_predict.json", "w")
        json.dump(data_out, out_file, indent = 6)    
    else:
        in_file = open("data_predict.json", "r")
        data_out = json.load(in_file) 
    return jsonify({'output':data_out})

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=8080)
