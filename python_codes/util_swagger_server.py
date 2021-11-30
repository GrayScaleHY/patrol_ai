from flask import Flask, jsonify, request
import json
from flask_restplus import Namespace, Resource, fields, reqparse, marshal
from flask_restplus import Api

flask_app = Flask(__name__)

app = Api(app = flask_app, 
          version = "1.0", 
		  title = "智能巡检算法接口", 
		  description = "提供巡检算法接口文档和简单的测试环境。")
# name_space = app.namespace('inspection')

bboxes = {"roi": fields.List(fields.Float, required = False, description="感兴趣区域roi。(可选)")}
bboxes = app.model('bboxes', bboxes)

pointers = {"center": fields.List(fields.Float, required = True, description="表盘中心点。（必须）")}
pointers = app.model('pointers', pointers, description="配置的点坐标， 至少要中心点、最大刻度、最小刻度这三个点。（必须）")

## 目标检测模板格式
config_object_detection = app.model('config_object_detection', {
	'img_ref': fields.String(required = False, description="模板图片，base64编码的图片字符串。(可选)"),
	'bboxes': fields.Nested(bboxes, required = True, description="框子信息")
})

config_pointer = app.model('config_pointer', {
	'img_ref': fields.String(required = True, description="模板图片，base64编码的图片字符串。(可选)"),
    "number": fields.Integer(required = True, description="总共有几个指针。(必须)"),
    'pointers': fields.Nested(pointers, required = True, description="配置的点坐标。(必选)"),
	'bboxes': fields.Nested(bboxes, required = True, description="框子信息。(必选)"),
    "dp": fields.Integer(required = False, description="结果保留的小数点位数。(可选，默认是3)"),
    "length": fields.Integer(required = False, description="指针长度，当有多个指针且可以根据指针长度来判断时需要填写。0表示最短，1表示中等，2表示最长。(可选)"),
    "width": fields.Integer(required = False, description="指针宽度，当有多个指针且可以根据指针宽度来判断时需要填写。0表示最宽，1表示中等，2表示最细。(可选)"),
    "color": fields.Integer(required = False, description="指针颜色，当有多个指针且可以根据指针颜色来判断时需要填写。0表示黑色，1表示白色，2表示红色。(可选)"),
})

data_object_detection = app.model('data', {
    'image': fields.String(required = True, description="base64编码的图片字符串。"),
    'config': fields.Nested(config_object_detection, required = True, description="模板信息"),
	'type': fields.String(required = True, description="巡检项目类型"),
    # how to add used features arrays
})

data_pointer = app.model('data_pointer', {
    'image': fields.String(required = True, description="base64编码的图片字符串。"),
    'config': fields.Nested(config_pointer, required = True, description="模板信息"),
	'type': fields.String(required = True, description="巡检项目类型"),
    # how to add used features arrays
})


### 仪表指针读数
@app.route('/inspection_pointer/')
class pointer_class(Resource):
    @app.expect(data_pointer)
    def post(self):
        """仪表指针读数识别"""
        input_data = json.loads(request.get_data(as_text=True))
        self.out_data = input_data
        return jsonify(self.out_data)


### 目标检测
@app.route('/inspection_pressplate/')
class pressplate_class(Resource):
    @app.expect(data_object_detection)
    def post(self):
        """压板状态识别接口"""
        input_data = json.loads(request.get_data(as_text=True))
        self.out_data = input_data
        return jsonify(self.out_data)

@app.route('/inspection_led/')
class led_class(Resource):
    @app.expect(data_object_detection)
    def post(self):
        """led灯状态识别接口"""
        input_data = json.loads(request.get_data(as_text=True))
        out_data = input_data
        return jsonify(out_data)
        
@app.route('/inspection_fire_smoke/')
class fire_smoke_class(Resource):
    @app.expect(data_object_detection)
    def post(self):
        """烟火状态识别接口"""
        input_data = json.loads(request.get_data(as_text=True))
        out_data = input_data
        return jsonify(out_data)

@app.route('/inspection_air_switch/')
class air_switch_class(Resource):
    @app.expect(data_object_detection)
    def post(self):
        """空气开关状态识别接口"""
        input_data = json.loads(request.get_data(as_text=True))
        out_data = input_data
        return jsonify(out_data)

@app.route('/inspection_fangpaiqi/')
class fangpaiqi_class(Resource):
    @app.expect(data_object_detection)
    def post(self):
        """翻牌器状态识别接口"""
        input_data = json.loads(request.get_data(as_text=True))
        out_data = input_data
        return jsonify(out_data)

@app.route('/inspection_helmet/')
class helmet_class(Resource):
    @app.expect(data_object_detection)
    def post(self):
        """安全帽佩戴状态识别接口"""
        input_data = json.loads(request.get_data(as_text=True))
        out_data = input_data
        return jsonify(out_data)

@app.route('/inspection_digital/')
class digital_class(Resource):
    @app.expect(data_object_detection)
    def post(self):
        """led数字识别接口"""
        input_data = json.loads(request.get_data(as_text=True))
        out_data = input_data
        return jsonify(out_data)

@app.route('/inspection_meter/')
class meter_class(Resource):
    @app.expect(data_object_detection)
    def post(self):
        """仪表定位识别接口"""
        input_data = json.loads(request.get_data(as_text=True))
        out_data = input_data
        return jsonify(out_data)

@app.route('/inspection_arrow/')
class arrow_class(Resource):
    @app.expect(data_object_detection)
    def post(self):
        """箭头仪表识别接口"""
        input_data = json.loads(request.get_data(as_text=True))
        out_data = input_data
        return jsonify(out_data)

@app.route('/inspection_rotary_switch/')
class rotary_switch_class(Resource):
    @app.expect(data_object_detection)
    def post(self):
        """切换把手状态识别接口"""
        input_data = json.loads(request.get_data(as_text=True))
        out_data = input_data
        return jsonify(out_data)




if __name__ == '__main__':
    flask_app.run(debug=True, port=5200, host='0.0.0.0')