from flask import Flask, request, jsonify
import model_predict
import json
import numpy as np
import time 

data_json = {'status':'','data_predict':[]}

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
    if(run == 1):
        forecast_flow = model_predict.run_prediction(type_feature=1,nb_past=past_history,nb_future=future_target,path_model=path_model_flow,url_get=url_get,mean_std=True)
        forecast_p = model_predict.run_prediction(type_feature=0,nb_past=past_history,nb_future=future_target,path_model=path_model_p,url_get=url_get,mean_std=True)
        json_dump = json.dumps({'forecast-flow':forecast_flow,
                                'forecast-Pressure':forecast_p},cls=NumpyEncoder)
        data_json['data_predict'] = json_dump
        named_tuple = time.localtime() 
        time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
        data_json['status'] = time_string   
    return jsonify({'output':data_json})

if __name__ == '__main__':
    app.run(debug=True,port=8080)
