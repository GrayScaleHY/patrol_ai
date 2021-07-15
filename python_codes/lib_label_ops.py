from label_ops.pascal_voc_io import PascalVocWriter, PascalVocReader
from label_ops.yolo_io import YoloReader, YOLOWriter
from PIL import Image
import os
import cv2
import labelme2coco # pip install labelme2coco
import xml.etree.ElementTree as ET
# from xml import etree


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

    for bbox in shapes:
        label = bbox[0]
        xmin = bbox[1][0][0]
        ymin = bbox[1][0][1]
        xmax = bbox[1][2][0]
        ymax = bbox[1][2][1]
        writer.addBndBox(xmin, ymin, xmax, ymax, label, 0)

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
    f_class = open(class_file, 'r')
    classes = f_class.read().strip('\n').split('\n')
    f_class.close()
    voc_info = PascalVocReader(xml_file)
    shapes = voc_info.getShapes()

    for bbox in shapes:
        name = bbox[0]
        if name in classes:
            label = classes.index(bbox[0])
            xmin = bbox[1][0][0]
            ymin = bbox[1][0][1]
            xmax = bbox[1][2][0]
            ymax = bbox[1][2][1]
            writer.addBndBox(xmin, ymin, xmax, ymax, label, 0)
        else:
            print("warning:", name, "not in classes")

    writer.save(targetFile=txt_file)


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

def labelme_2_coco(labelme_folder, save_json_path):
    """
    labelme标注的目标分割数据转为coco格式
    https://github.com/fcakyon/labelme2coco
    注意：
        save_json_path不能在labelme_folder中，
        最好是labelme_folder和img在同一个目录。
    """
    labelme2coco.convert(labelme_folder, save_json_path)


if __name__ == '__main__':
    # img_file = "C:/data/meter/test1/2021_4_27_meter_122.jpg"
    # txt_file = "C:/data/meter/test1/2021_4_27_meter_122.txt"
    xml_raw = "C:/data/meter/test1/num/2021_4_27_meter_122.xml"
    xml_part = "C:/data/meter/test1/meter/2021_4_27_meter_122.xml"
    # class_file = "C:/data/meter/test1/classes.txt"


