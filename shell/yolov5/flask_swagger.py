from flask import Flask, request
from flask_restplus import Api, Resource, fields
import time
import os

flask_app = Flask(__name__)
app = Api(app = flask_app, 
		  version = "1.0", 
		  title = "两库一台算法接口", 
		  description = "两库一台的算法接口，包括模型训练和模型测试")

gpu_01 = app.namespace('gpu_01', description='训练专用')

gpu_02 = app.namespace('gpu_02', description='测试专用')

list_of_names = {}

data_info = {"finish": 1, "saved_path": "", "train_rate": ""}

def train(model_name, epoch):
    data_info["finish"] = 0
    now = time.strftime("%m-%d-%H-%M-%S")
    data_info["saved_path"] = "/data/yolov5/%s/saved_model/%s/bast.pt"%(now, epoch)
    for i in range(epoch):
        time.sleep(5)
        # if i == 2:
        #     print(now + 1)
        data_info["train_rate"] = str(i+1) + "/" + str(epoch)
    data_info["finish"] = 1

@gpu_01.route("/train/<string:model_name>,<int:epoch>")
class MainTrain(Resource):
    @app.doc(responses={ 200: 'OK', 400: 'Invalid Argument', 500: 'Mapping Key Error' }, 
                params={'model_name': '待训练模型的名字，例如 led, fangpaiqi',
                        'epoch': "总训练批次"})
    # @app.expect(model)		
    def post(self, model_name, epoch):
        """启动训练的接口"""
        try:
            train(model_name, epoch)
            return data_info
        except KeyError as e:
            gpu_01.abort(500, e.__doc__, status = "Could not save information", statusCode = "500")
        except Exception as e:
            gpu_01.abort(400, e.__doc__, status = "Could not save information", statusCode = "400")

@gpu_01.route("/info/")
class MainInfo(Resource):
    @app.doc(responses={ 200: 'OK', 400: 'Invalid Argument', 500: 'Mapping Key Error' })
    def get(self):
        """获取当前训练信息的接口"""
        try:
            return data_info
        except KeyError as e:
            gpu_01.abort(500, e.__doc__, status = "Could not retrieve information", statusCode = "500")
        except Exception as e:
            gpu_01.abort(400, e.__doc__, status = "Could not retrieve information", statusCode = "400")

@gpu_02.route("/detect/<string:model_name>,<string:image_name>")
class MainDetect(Resource):
    @app.doc(responses={ 200: 'OK', 400: 'Invalid Argument', 500: 'Mapping Key Error' }, 
                params={'model_name': '模型名字', "image_name": "图片名字"})
    # @app.expect(model)		
    def post(self, model_name, image_name):
        """启动测试的接口"""
        try:
            if os.path.isdir(image_name):
                return {"save_dir": image_name + "_result"}
            else:
                return {"warning": "image_name is not exist !"}
        except KeyError as e:
            gpu_01.abort(500, e.__doc__, status = "Could not save information", statusCode = "500")
        except Exception as e:
            gpu_01.abort(400, e.__doc__, status = "Could not save information", statusCode = "400")

if __name__ == '__main__':
    flask_app.run(debug=True, port=5200, host='0.0.0.0')