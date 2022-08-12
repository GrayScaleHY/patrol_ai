from label_ops.pascal_voc_io import PascalVocWriter, PascalVocReader
from label_ops.yolo_io import YoloReader, YOLOWriter
from PIL import Image
import os
import cv2
# import labelme2coco # pip install labelme2coco
import numpy as np
from lib_image_ops import img2base64
import json
# import xmltodict
# from xml import etree
import xml.etree.ElementTree as ET

def convert_rec(img, box):
    """
    args:
        img: img_data
        box: [ox, oy, w, h] 或者 [xmin, ymin, xmax, ymax]
    return:
        与输入的box格式相反。
    """
    H, W = img.shape[:2]

    ## [ox, oy, w, x] -> [xmin, ymin, xmax, ymax]
    if 0 < float(box[0]) < 1:
        ox = box[0] * W; oy = box[1] * H
        w = box[2] * W; h = box[3] * H
        xmin = int(ox - w / 2)
        ymin = int(oy - h / 2)
        xmax = int(ox + w / 2)
        ymax = int(oy + h / 2)
        return [xmin, ymin, xmax, ymax]

    ## [xmin, ymin, xmax, ymax] -> [ox, oy, w, x]
    else:
        ox = (box[0] + box[2]) / (2 * W)
        oy = (box[1] + box[3]) / (2 * H)
        w = abs(box[2] - box[0]) / W
        h = abs(box[3] - box[1]) / H
        return [ox, oy, w, h]

def get_xml_cfgs(xml_file):
    """
    获取voc格式xml文件中的标签信息
    args:
        xml_file: xml 文件
    return:
        cfgs: 标签信息，格式为 格式为[{"label": "", "coor": [xmin, ymin, xmax, ymax]}, ..)
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size') # 图片大小
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    cfgs = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        if int(difficult) == 1:
            continue

        label = obj.find('name').text # label名字

        xmlbox = obj.find('bndbox') # 框子坐标
        b = [float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), 
            float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text)]

        if 0 <= b[0] <= b[2] <= w and 0 <= b[1] <= b[3] <= h: # 判断框子是否超出边界
            cfgs.append({"label": label, "coor": b})
            
    return cfgs

def convert_points(points, bbox):
    """
    判断points是否都在bbox中，如果在则返回相对与bbox的points，否则，返回None.
    args:
        points: [[x0, y0], [x1, y1], ...]
        bbox: [xmin, ymin, xmax, ymax]
    return:
        points or None
    """
    points = np.array(points, dtype=float)
    a = points[:, 0]
    if bbox[0] > np.min(points[:, 0]) or bbox[2] < np.max(points[:, 0]):
        return None
    if bbox[1] > np.min(points[:, 1]) or bbox[3] < np.max(points[:, 1]):
        return None
    for i in range(len(points)):
        points[i][0] -= bbox[0]
        points[i][1] -= bbox[1]
        
    return points.tolist()


def rel_coordinates(ref_coo, tag_coo):
    """
    坐标转换。
    return:
        False表示：tag_coo == ref_coo 或tag_coo不完全被ref_coo包含
        [xmin, ymin, xmax, ymax]: 相对于ref_coo的坐标
    """
    assert len(ref_coo) == len(tag_coo) == 4, "维度不匹配。"
    if ref_coo == tag_coo:
        return False
    else:
        if ref_coo[0] > tag_coo[0] or ref_coo[1] > tag_coo[1]:
            return False
        elif ref_coo[2] < tag_coo[2] or ref_coo[3] < tag_coo[3]:
            return False
        else:
            xmin = tag_coo[0] - ref_coo[0]
            ymin = tag_coo[1] - ref_coo[1]
            xmax = tag_coo[2] - ref_coo[0]
            ymax = tag_coo[3] - ref_coo[1]
            return [xmin, ymin, xmax, ymax]


def crop_img_base_label(img_file, xml_file, save_dir, crop_label):
    """
    根据xml标签信息裁剪图片。
    若裁剪部分还完全包含别的目标，则会生成一个相对与裁剪图片的xml标签文件。
    args:
        img_file: 图片路径
        xml_file: 标签路径
        save_dir: 保存目录(裁剪后的图片和裁剪后的标签)
        crop_label: 需要裁剪的label
    """
    img_cv = cv2.imread(img_file) # 读取图片
    ## 路径中有中文字符是可使用该方法读取图片
    # img_cv = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), -1)

    ## 获取xml_file中的目标框信息
    voc_info = PascalVocReader(xml_file)
    bboxs = voc_info.getShapes()

    count = 0
    for ref_bbox in bboxs:  # 需要裁剪的目标bbox
        ref_label = ref_bbox[0]
        if crop_label == ref_label:
            ref_xmin = ref_bbox[1][0][0]
            ref_ymin = ref_bbox[1][0][1]
            ref_xmax = ref_bbox[1][2][0]
            ref_ymax = ref_bbox[1][2][1]
            obj_img = img_cv[int(ref_ymin):int(ref_ymax), int(ref_xmin):int(ref_xmax)]
            img_name = os.path.basename(img_file)
            img_out = os.path.join(save_dir, img_name[:-4] + "_" +str(count) + "_" + ref_label + ".jpg")
            xml_out = img_out[:-4] + ".xml"
            cv2.imwrite(img_out ,obj_img)  # 保存裁剪图片
            count += 1

            bbox_list = []
            for tag_bbox in bboxs:
                tag_label = tag_bbox[0]
                tag_xmin = tag_bbox[1][0][0]
                tag_ymin = tag_bbox[1][0][1]
                tag_xmax = tag_bbox[1][2][0]
                tag_ymax = tag_bbox[1][2][1]

                ref_coo = [ref_xmin, ref_ymin, ref_xmax, ref_ymax]
                tag_coo = [tag_xmin, tag_ymin, tag_xmax, tag_ymax]
                tag_coo = rel_coordinates(ref_coo, tag_coo)  #坐标转换
                # print(tag_coo)
                if tag_coo:
                    bbox_list.append([tag_label] + tag_coo)
            
            if len(bbox_list) != 0:
                out_name = os.path.basename(img_out)
                folder = os.path.basename(os.path.dirname(img_out))
                img_size = [ref_ymax-ref_ymin, ref_xmax-ref_xmin, 3] # [h, w, d]
                writer = PascalVocWriter(folder, out_name, img_size, localImgPath=img_out)

                for bbox in bbox_list:# [label, xmin, ymin, xmax, ymax]
                    writer.addBndBox(bbox[1], bbox[2], bbox[3], bbox[4], bbox[0], 0)

                writer.save(targetFile=xml_out)


def crop_img_base_json(img_file, xml_file, json_file, save_dir):
    """
    根据目标框截取图片，并且将图像分割标注信息保留。
    """
    os.makedirs(save_dir, exist_ok=True)

    ## 图片信息
    img = cv2.imread(img_file)

    ## json文件信息
    f = open(json_file, 'r', encoding='utf-8')
    labelme_info = json.load(f)
    f.close()

    ## 获取xml_file中的目标框信息
    voc_info = PascalVocReader(xml_file)
    bboxs = voc_info.getShapes()

    count = 0
    for b in bboxs:  # 需要裁剪的目标bbox
        out_info = {}
        bbox = [b[1][0][0], b[1][0][1], b[1][2][0], b[1][2][1]]
        img_box = img[bbox[1]:bbox[3], bbox[0]: bbox[2]]

        # json文件的值
        version = labelme_info["version"]
        flags = labelme_info["flags"]
        imagePath = os.path.join(save_dir, os.path.basename(img_file)[:-4]+"_"+str(count)+".jpg")
        
        imageData = img2base64(img_box)
        imageHeight = img_box.shape[0]
        imageWidth = img_box.shape[1]
        shapes = []

        ## 逐个分析shape
        for i, shape in enumerate(labelme_info["shapes"]):
            points = shape["points"]
            points = convert_points(points, bbox) # 将坐标点转为相对与bbox的坐标值。
            if points is not None:
                label = shape["label"]
                group_id = shape["group_id"]
                shape_type = shape["shape_type"]
                flags = shape["flags"]
                shape = {"label":label,"points":points,"group_id":group_id,"shape_type":shape_type,"flags":flags}
                shapes.append(shape)
        if len(shapes) > 0:
            count += 1
            out_info["version"] = version
            out_info["flags"] = flags
            out_info["shapes"] = shapes
            out_info["imagePath"] = imagePath
            out_info["imageData"] = imageData
            out_info["imageHeight"] = imageHeight
            out_info["imageWidth"] = imageWidth

            ## 保存
            cv2.imwrite(imagePath, img_box)
            f = open(imagePath[:-4] + ".json", "w")
            json.dump(out_info, f, ensure_ascii=False, indent=2)
            f.close()
            print(imagePath)

            
def yolo2voc(img_file, txt_file, xml_file, class_file):
    """
    yolo标注格式转voc格式。
    """
    image = Image.open(img_file)
    img_size = [image.size[1], image.size[0], 3]
    folder = os.path.basename(os.path.dirname(img_file))
    img_name = os.path.basename(img_file)
    writer = PascalVocWriter(folder, img_name, img_size, localImgPath=img_file)
    yolo_info = YoloReader(txt_file, img_size, classListPath=class_file)
    shapes = yolo_info.getShapes()  # 获取bbox的信息

    count = 0
    for bbox in shapes:
        label = bbox[0]
        xmin = bbox[1][0][0]
        ymin = bbox[1][0][1]
        xmax = bbox[1][2][0]
        ymax = bbox[1][2][1]
        writer.addBndBox(xmin, ymin, xmax, ymax, label, 0)
        count += 1
    
    if count > 0:
        writer.save(targetFile=xml_file)


def voc2yolo(img_file, xml_file, txt_file, class_file):
    """
    voc标注格式转yolo标注格式。
    """
    image = Image.open(img_file)
    img_size = [image.size[1], image.size[0], 3]
    folder = os.path.basename(os.path.dirname(img_file))
    img_name = os.path.basename(img_file)
    writer = YOLOWriter(folder, img_name, img_size, localImgPath=img_file)
    # Read classes.txt
    f_class = open(class_file, 'r', encoding='utf-8')
    classes = f_class.read().strip('\n').split('\n')
    f_class.close()
    voc_info = PascalVocReader(xml_file)
    shapes = voc_info.getShapes()

    count = 0
    for bbox in shapes:
        name = bbox[0]
        if name in classes:
            label = classes.index(bbox[0])
            xmin = bbox[1][0][0]
            ymin = bbox[1][0][1]
            xmax = bbox[1][2][0]
            ymax = bbox[1][2][1]
            writer.addBndBox(xmin, ymin, xmax, ymax, label, 0)
            count += 1
        else:
            print("warning:", name, "not in classes")

    if count > 0:
        writer.save(targetFile=txt_file)


def cvat2labelme(img_file, xml_file, json_file):
    """
    将cvat的labelme的xml格式的标签文件转成labelme软件保存的json格式标签文件。
    """
    ## 读取xml文件，转成dict格式
    f = open(xml_file, 'r',encoding='utf-8')
    xml_str = f.read() #读取xml文件内容
    dict_ = xmltodict.parse(xml_str) #将读取的xml内容转为dict
    f.close()

    shapes = []

    if "object" not in dict_["annotation"]:
        return 0

    objects = dict_["annotation"]["object"]
    if not isinstance(objects, list):
        objects = [objects]
    for obj in objects:
        label = obj["name"]
        points = [[float(point["x"]), float(point["y"])] for point in obj["polygon"]["pt"]]
        shapes.append({"label": label, "points": points, "group_id": None, "shape_type": "polygon", "flags": {}})

    imagePath = os.path.basename(img_file)
    img = cv2.imread(img_file)
    imageData = img2base64(img)
    imageHeight = img.shape[0]
    imageWidth = img.shape[1]

    obj_json = {"version": "4.5.6",
                "flags": {},
                "shapes": shapes,
                "imagePath": imagePath,
                "imageData": imageData,
                "imageHeight": imageHeight,
                "imageWidth": imageWidth}

    f = open(json_file, "w", encoding='utf-8')
    json.dump(obj_json, f, indent=2, ensure_ascii=False) # 保存json
    f.close()


def xml_merge(xml_raw, xml_part):
    """
    将两个xml文件合并成一个。
    """
    voc_info_raw = PascalVocReader(xml_raw)
    voc_info_part = PascalVocReader(xml_part)
    img_name = voc_info_raw.get_infos()["filename"]
    folder = voc_info_raw.get_infos()["folder"]
    img_size = voc_info_raw.get_infos()["size"]
    img_file = voc_info_raw.get_infos()["path"]

    assert  img_name== voc_info_part.get_infos()["filename"], xml_raw + \
        "和" + xml_part + ": 两个xml文件的filename不相同！"
    assert img_size == voc_info_part.get_infos()["size"], xml_raw + \
        "和" + xml_part + ": 两个xml文件的size不相同！"

    bbox_names = []
    writer = PascalVocWriter(folder, img_name, img_size, localImgPath=img_file)
    for bbox in voc_info_raw.getShapes():
        label = bbox[0]
        xmin = bbox[1][0][0]
        ymin = bbox[1][0][1]
        xmax = bbox[1][2][0]
        ymax = bbox[1][2][1]
        writer.addBndBox(xmin, ymin, xmax, ymax, label, 0)
        bbox_names.append(label)

    for bbox in voc_info_part.getShapes():
        label = bbox[0]
        if label not in bbox_names:
            xmin = bbox[1][0][0]
            ymin = bbox[1][0][1]
            xmax = bbox[1][2][0]
            ymax = bbox[1][2][1]
            writer.addBndBox(xmin, ymin, xmax, ymax, label, 0)

    writer.save(targetFile=xml_raw)


def labelme_2_coco(labelme_folder, coco_json_file):
    """
    labelme标注的目标分割数据转为coco格式
    https://github.com/fcakyon/labelme2coco
    注意：
        coco_json_file不能在labelme_folder中，
        最好是labelme_folder和img在同一个目录。
    """
    labelme2coco.convert(labelme_folder, coco_json_file)


def int_bndbox(xml_file):
    """
    将xml文件中的bndbox坐标值改为int型,并保存回源路径
    return:
        文件是否发生了转换，如果转换了，返回True， 否则，返回False.
    """

    if not xml_file.endswith(".xml"): # 必须是.xml结尾的文件
        return False
    
    ## 将xml文件内容转为dict
    f = open(xml_file, "r", encoding='utf-8')
    content = f.read()
    f.close()
    dict_ = xmltodict.parse(content)

    try:
        objects = dict_['annotation']['object'] #
    except:
        return False

    if isinstance(objects, list):
        for i, object in enumerate(objects):
            try:
                ## 将bndbox中的float型数据改为int
                objects[i]['bndbox']['xmin'] = int(float(object['bndbox']['xmin']))
                objects[i]['bndbox']['ymin'] = int(float(object['bndbox']['ymin']))
                objects[i]['bndbox']['xmax'] = int(float(object['bndbox']['xmax']))
                objects[i]['bndbox']['ymax'] = int(float(object['bndbox']['ymax']))
            except:
                return False
    else:
        try:
            ## 将bndbox中的float型数据改为int
            objects['bndbox']['xmin'] = int(float(objects['bndbox']['xmin']))
            objects['bndbox']['ymin'] = int(float(objects['bndbox']['ymin']))
            objects['bndbox']['xmax'] = int(float(objects['bndbox']['xmax']))
            objects['bndbox']['ymax'] = int(float(objects['bndbox']['ymax']))
        except:
            return False
    
    dict_['annotation']['object'] = objects # 改变后的dict

    ## 新的dict重新保存成xml文件
    xml_res = xmltodict.unparse(dict_, pretty=True)
    f = open(xml_file, "w", encoding='utf-8')
    f.write(xml_res)
    f.close()

    return True


def int_bndbox_batch(dir):
    """
    将文件夹中的xml文件中的bndbox坐标值改为int型
    """
    count = 0
    for root, dirs, files in os.walk(dir):
        for file_name in files:
            if not file_name.endswith(".xml"):
                continue
            xml_file = os.path.join(root, file_name)
            is_conv = int_bndbox(xml_file)
            if is_conv:
                count += 1
    print("convert xml num:", count)


if __name__ == '__main__':
    import glob
    import os

    class_file = "C:/Users/yuanhui/Desktop/hear/test/test/obj.names"
    for jpg_file in glob.glob("C:/Users/yuanhui/Desktop/hear/test/test/*.jpg"):
        xml_file = jpg_file[:-4] + ".xml"
        txt_file = jpg_file[:-4] + ".txt"
        voc2yolo(jpg_file, xml_file, txt_file, class_file)
