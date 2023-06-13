
import numpy as np
import cv2
import os
try:
    from cnocr import CnOcr
    ocr = CnOcr()  # 所有参数都使用默认值
    ocr_type = "cnocr"
except:
    if os.path.exists("ppocr"):
        os.rename("ppocr", "ppocr_old")
    from paddleocr import PaddleOCR # pip install paddleocr
    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
    ocr_type = "ppocr"

def inference_ocr(img):
    """
    ocr推理代码
    args:
        img: image data
    return:
        result: 结构: [{"content": txt, "bbox": bbox, "score": score}, ..]
    """
    if ocr_type == "cnocr":
        return inference_cnocr(img)
    else:
        return inference_ppocr(img)
    

def inference_cnocr(img):
    """
    cnocr
    使用方法：https://pypi.org/project/cnocr/
    下载模型：https://huggingface.co/breezedeus/cnstd-cnocr-models/tree/main/models
    args:
        img: image data
    return:
        result: 识别结果，包括content, bbox, score
                结构: [{"content": txt, "bbox": bbox, "score": score}, ..]
    """
    res_raw = ocr.ocr(img)
    result = []
    for res in res_raw:
        content = res["text"]
        score = res["score"]
        coors = res["position"]
        box = [int(min(coors[:,0])), int(min(coors[:,1])), int(max(coors[:, 0])), int(max(coors[:, 1]))]
        result.append({"content": content, "bbox": box, "score": score})
    return result


def inference_ppocr(img):
    """
    paddle paddle ocr 的推理方法。
    https://github.com/PaddlePaddle/PaddleOCR
    https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/quickstart.md

    安装paddlepaddle: https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html
    安装paddleocr: pip install "paddleocr>=2.0.1" -i https://mirror.baidu.com/pypi/simple
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
    img_path = '/home/yh/yolov5/runs/detect/exp2/shandongqj_052514_0009.jpg'
    img = cv2.imread(img_path)
    for i in range(5):
        start = time.time()
        result = inference_cnocr(img)
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
