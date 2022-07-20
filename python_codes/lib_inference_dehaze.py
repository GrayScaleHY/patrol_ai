import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize, ToPILImage
from psd_models.FFA import FFANet
from psd_models.MSBDN import MSBDNNet
import cv2
import numpy as np

def load_psd_model(mdl_file):
    """
    加载dehaze_psd模型，PSB-MSBDN或者PSD-FFANET，返回net
    https://github.com/zychen-ustc/PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors
    """
    # net = FFANet(3, 19)
    net = MSBDNNet()
    net = nn.DataParallel(net, device_ids=[0])
    net.load_state_dict(torch.load(mdl_file))
    net.eval()
    return net


def inference_psd(net, img):
    """
    psd模型推理。
    https://github.com/zychen-ustc/PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors
    args:
        net: 加载的去雾模型，用load_psd_model加载
        img: image data
    return:
        img: 去雾后的image data
    """
    haze_img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) # cv2转Image  .convert('RGB')
    # haze_img = Image.open(img_file).convert('RGB') # 图片
    haze_reshaped = haze_img.resize((512, 512), Image.ANTIALIAS)

    ## Image转tensor，并且添加一维
    transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    haze = transform_haze(haze_img).unsqueeze(0)  # unsqueeze表示添加维度
    haze_reshaped = transform_haze(haze_reshaped).unsqueeze(0)

    _, pred, T, A, I = net(haze, haze_reshaped, True) # inference

    ts = pred.squeeze(0) # 去掉一维

    # image = ToPILImage()(ts) # tensor转Image
    # image1 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) # Image转cv2

    ## tensor转cv2
    ts = ts.to(torch.device('cpu'))
    img = ts.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


if __name__ == '__main__':
    import time
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    img_file = "/home/yh/PSD/images/test1.jpg"
    out_file = "test_res.jpg"
    mdl_file = '/data/PatrolAi/dehaze_psd/PSB-MSBDN'
    img = cv2.imread(img_file)
    net = load_psd_model(mdl_file)
    start = time.time()
    img = inference_psd(net, img)
    print(time.time() - start)
    cv2.imwrite(out_file,img)
