from label_ops.pascal_voc_io import PascalVocWriter, PascalVocReader
from label_ops.yolo_io import YoloReader, YOLOWriter
from PIL import Image
import os
from PIL import Image
import labelme2coco # pip install labelme2coco


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
        x_max = bbox[1][2][0]
        y_max = bbox[1][2][1]
        writer.addBndBox(xmin, ymin, x_max, y_max, label, 0)

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
            x_max = bbox[1][2][0]
            y_max = bbox[1][2][1]
            writer.addBndBox(xmin, ymin, x_max, y_max, label, 0)
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
        x_max = bbox[1][2][0]
        y_max = bbox[1][2][1]
        writer.addBndBox(xmin, ymin, x_max, y_max, label, 0)
        bbox_names.append(label)

    for bbox in voc_info_part.getShapes():
        label = bbox[0]
        if label not in bbox_names:
            xmin = bbox[1][0][0]
            ymin = bbox[1][0][1]
            x_max = bbox[1][2][0]
            y_max = bbox[1][2][1]
            writer.addBndBox(xmin, ymin, x_max, y_max, label, 0)

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


