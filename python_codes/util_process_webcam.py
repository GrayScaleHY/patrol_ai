import cv2, queue, threading, time
import json
import requests
from lib_image_ops import base642img, img2base64

# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

## 请求服务的config
API = "http://192.168.57.159:5000/inspection_disconnector/"
close_file = "C:/data/disconnector/test_close.jpg"
open_file = "C:/data/disconnector/test_open.jpg"
img_open = cv2.imread(open_file)
img_close = cv2.imread(close_file)
config = {"img_open": img2base64(img_open), "img_close": img2base64(img_close), "bbox": [603, 218, 707, 342]}

cap = VideoCapture('rtsp://admin:ut0000@192.168.57.25') # 读取网络相机 'rtsp://admin:ut0000@192.168.57.25'
while True:
    # time.sleep(2)   # simulate time between events
    start = time.time()
    frame = cap.read() #读取最新帧

    ## 处理图片
    img_tag = img2base64(frame)
    input_data = {"image": img_tag, "config": config, "type": "disconnector"}
    send_data = json.dumps(input_data)
    res = requests.post(url=API, data=send_data).json()
    img_base64 = res["img_result"]
    if len(img_base64) > 10:
        img = base642img(img_base64)
    else:
        img = frame

    cv2.imshow("result", img)
    if chr(cv2.waitKey(1)&255) == 'q':
        break

    print("spend time:", time.time()-start)




    





