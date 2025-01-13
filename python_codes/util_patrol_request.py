import json
import requests
from flask import request,make_response 
import time
from lib_help_base import traverse_and_modify

API = "http://192.168.44.135:29528/inspection_pointer/"

json_file = "input_data.json"
f = open(json_file,"r",encoding='utf-8')
input_data = json.load(f)
f.close()
send_data = json.dumps(input_data)

start_time = time.time()

res = requests.post(url=API, data=send_data).json()

print("spend time:", round(time.time() - start_time, 3), "out data:", traverse_and_modify(res))