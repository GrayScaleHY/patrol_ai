"""
paddle paddle ocr 的推理方法。
https://github.com/PaddlePaddle/PaddleOCR
github源目录下的tools改名问ppocr_tools,为了避免detectron2冲突。

安装paddlepaddle: https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html
安装paddleocr: pip install "paddleocr>=2.0.1" -i https://mirror.baidu.com/pypi/simple
"""

import ppocr_tools.infer.utility as utility
from ppocr_tools.infer.predict_system import TextSystem
import cv2
import time

def load_ppocr(det_model_dir, cls_model_dir, rec_model_dir):
    """
    加载ppocr模型。分为检测模型(det_model_dir)， 分类模型(cls_model_dir), 识别模型(rec_model_dir).
    """
    args = utility.parse_args()
    args.det_model_dir = det_model_dir
    args.cls_model_dir = cls_model_dir
    args.rec_model_dir = rec_model_dir
    args.use_angle_cls = "true"
    text_sys = TextSystem(args)
    return text_sys


def inference_ppocr(img, text_sys):
    """
    使用ocr推理。
    args:
        img: image data
        text_sys: 加载的ocr模型。使用load_ppocr加载。
    return:
        result: 识别结果，包括content, bbox, score
                结构: [{"content": txt, "bbox": bbox, "score": score}, ..]
    """
    dt_boxes, rec_res = text_sys(img)
    result = []
    for i, coors in enumerate(dt_boxes):
        txt = rec_res[i][0]
        score = rec_res[i][1]
        xs = [a[0] for a in coors]
        ys = [a[1] for a in coors]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        result.append({"content": txt, "bbox": bbox, "score": score})
    return result


if __name__ == "__main__":

    import glob
    from lib_image_ops import img_chinese

    det_model_dir = "/data/inspection/ppocr/ch_PP-OCRv2_det_infer/"
    cls_model_dir = "/data/inspection/ppocr/ch_ppocr_mobile_v2.0_cls_infer/"
    rec_model_dir = "/data/inspection/ppocr/ch_PP-OCRv2_rec_infer/"
    text_sys = load_ppocr(det_model_dir, cls_model_dir, rec_model_dir)

    image_file = "test/img_tag.jpg"
    img = cv2.imread(image_file)
    result = inference_ppocr(img, text_sys)
    print("------------------------------")
    print(image_file)
    print(result)
    for out_data in result:
        bbox = out_data["bbox"]
        content = out_data["content"]
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])), (0, 0, 255), thickness=2)
        img = img_chinese(img, content, (int(bbox[0]), int(bbox[1]-20)),(0, 0, 255), size=20)
    cv2.imwrite(image_file[:-4] + "_result.jpg", img)





