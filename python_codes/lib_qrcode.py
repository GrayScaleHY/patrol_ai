import cv2
from pyzbar.pyzbar import decode # sudo apt-get install libzbar-dev; pip install pyzbar

def decoder(img):
    """
    二维码定位，并且读取二维码信息
    https://towardsdatascience.com/build-your-own-barcode-and-qrcode-scanner-using-python-8b46971e719e
    args:
        img: image data
    return:
        info: 格式为: [{'bbox': [xmin,ymin,xmax,ymax], 'content':content, "c_type":c_type}, ..]
    """
    ## 彩图变灰度图
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    barcode = decode(img) # 解二维码

    info = []
    for obj in barcode:
        ## 取二维码的四边形框
        # points = obj.polygon
        # pts = np.array(points, np.int32)
        # pts = pts.reshape((-1, 1, 2))
        # cv2.polylines(img, [pts], True, (0, 255, 0), 3)

        ## 取二维码的矩形框
        (x,y,w,h) = obj.rect
        bbox = [float(x), float(y), float(x+w), float(y+h)]
        
        ## 取二维码的内容合类型
        content = str(obj.data.decode("utf-8"))
        c_type = str(obj.type)

        info.append({"bbox": bbox, "content": content, "c_type": c_type})

    return info

if __name__ == '__main__':
    img = cv2.imread("/home/yh/image/python_codes/test/img_tag.jpg")
    info = decoder(img)
    print(info)