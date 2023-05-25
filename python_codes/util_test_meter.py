from app_pointer import pointer_detect
import os
import cv2

img_dir = "images"

for img_name in os.listdir(img_dir):
    img_file = os.path.join(img_dir, img_name)
    img = cv2.imread(img_file)

    seg_cfgs, roi_tag = pointer_detect(img, 10)

    for seg_cfg in seg_cfgs:
        seg = seg_cfg["seg"]
        cv2.line(img, (int(seg[0]), int(seg[1])),(int(seg[2]), int(seg[3])), (0, 255, 0), 2)
    
    c = roi_tag
    cv2.rectangle(img, (int(c[0]), int(c[1])),(int(c[2]), int(c[3])), (255,0,255), thickness=1)

    cv2.imwrite(img_file[:-4] + ".jpg", img)



