from yaml import load
from lib_inference_yolov5 import load_yolov5_model, inference_yolov5
import cv2
import glob

model = load_yolov5_model("/data/PatrolAi/yolov5/18cls_rec_defect_x6.pt")
img_file = "/data/PatrolAi/result_patrol/20221115_nc1.bmp"

for img_file in glob.glob("/data/PatrolAi/result_patrol/*.bmp"):
    img = cv2.imread(img_file)
    cfgs = inference_yolov5(model, img, resize=1280, conf_thres=0.2, iou_thres=0.2, pre_labels=None)
    print(cfgs)
    for cfg in cfgs:
        c = cfg["coor"]
        label = cfg["label"]
        cv2.rectangle(img, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), (255,0,255), thickness=2)
        cv2.putText(img, label, (int(c[0]), int(c[1])-5),cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), thickness=4)

    cv2.imwrite(img_file[:-4] + "_result.jpg", img)
