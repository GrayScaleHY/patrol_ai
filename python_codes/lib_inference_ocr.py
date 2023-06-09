"""
paddle paddle ocr 的推理方法。
https://github.com/PaddlePaddle/PaddleOCR
https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/quickstart.md

安装paddlepaddle: https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html
安装paddleocr: pip install "paddleocr>=2.0.1" -i https://mirror.baidu.com/pypi/simple
"""

import numpy as np
import cv2
import os
if os.path.exists("ppocr"):
    os.rename("ppocr", "ppocr_old")
from paddleocr import PaddleOCR # pip install paddleocr

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory

def inference_ppocr(img):
    """
    使用ocr推理。
    args:
        img: image data
    return:
        result: 识别结果，包括content, bbox, score
                结构: [{"content": txt, "bbox": bbox, "score": score}, ..]
    """
    res_raw = ocr.ocr(img, cls=True)[0]
    result = []
    for res in res_raw:
        content = res[1][0]
        score = res[1][1]
        coors = np.array(res[0])
        box = [int(min(coors[:,0])), int(min(coors[:,1])), int(max(coors[:, 0])), int(max(coors[:, 1]))]
        result.append({"content": content, "bbox": box, "score": score})
    return result

if __name__ == '__main__':
    from lib_image_ops import img_chinese
    import time
    img_path = '/data/PatrolAi/result_patrol/#0181_org_0_meter.jpg'
    img = cv2.imread(img_path)
    for i in range(5):
        start = time.time()
        result = inference_ppocr(img)
        print("infer spend time:", time.time() - start)
    print("------------------------------")
    print(result)
    for out_data in result:
        bbox = out_data["bbox"]
        content = out_data["content"]
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])), (0, 0, 255), thickness=2)
        img = img_chinese(img, content, (int(bbox[0]), int(bbox[1]-20)),(0, 0, 255), size=20)
    cv2.imwrite(img_path[:-4] + "_result.jpg", img)
