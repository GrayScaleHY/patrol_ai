from lib_inference_mrcnn import load_maskrcnn_model, inference_maskrcnn
from lib_inference_yolov5 import load_yolov5_model, inference_yolov5
import cv2
import os
import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

yolo_model = load_yolov5_model("/data/inspection/yolov5/meter.pt")
msrcnn_model = load_maskrcnn_model("/data/inspection/maskrcnn/pointer.pth")
img_file = "test/test.jpg"

dir_names = ['11-19-10-39-05', '11-19-10-39-07', '11-19-10-39-09', '11-19-10-39-12', '11-19-10-39-14', '11-19-10-39-17', '11-19-10-39-19', '11-19-10-39-22', '11-19-10-39-24', '11-19-10-39-27', '11-19-10-39-29', '11-19-10-39-32', '11-19-10-39-34', '11-19-10-39-37', '11-19-10-39-39', '11-19-10-39-41', '11-19-10-39-44', '11-19-10-39-47', '11-19-10-39-49', '11-19-10-39-52']

for img_file in glob.glob("/home/yh/image/python_codes/inspection_result/*.png"):
# for dir_name in dir_names:
    # img_file = "/home/yh/image/python_codes/inspection_result/pointer/11-17-16-34-34/img_tag.jpg"
    # img_file = os.path.join("/home/yh/image/python_codes/inspection_result/pointer", dir_name, "img_tag.jpg")

    img = cv2.imread(img_file)
    bbox_cfg = inference_yolov5(yolo_model, img, resize=640)
    img_draw = img.copy()

    for cfg in bbox_cfg:

        c = cfg["coor"]
        img_meter = img[int(c[1]):int(c[3]), int(c[0]):int(c[2])]

        cv2.rectangle(img_draw, (int(c[0]), int(c[1])),
                        (int(c[2]), int(c[3])), (0, 0, 255), thickness=2)

        contours, boxes, _ = inference_maskrcnn(msrcnn_model, img_meter)
        print(boxes)

        for box in boxes:

            box = [box[0]+c[0], box[1]+c[1], box[2]+c[0], box[3]+c[1]]

            cv2.rectangle(img_draw, (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])), (0, 255, 0), thickness=2)

    cv2.imwrite(img_file[:-4] + "_result.jpg", img_draw)



