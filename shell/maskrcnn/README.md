### maskrcnn教程
该教程讲述基于detectron2的maskrcnn分割模型的数据处理、模型训练、推理的流程。 
相关参考资料：[详细demo](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=QHnVupBBn9eR), [使用文档](https://detectron2.readthedocs.io/en/latest/tutorials/index.html), [github](https://github.com/facebookresearch/detectron2)
##### 1. 数据处理
detectron2的训练数据标签格式参考[说明文档](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#standard-dataset-dicts), [demo](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=b2bjrfb2LDeo)。也可根据下面步骤迅速制作简单的可训练数据集。  
(1). 使用CVAT对目标镜进行分割标注,导出标签文件(.xml结尾)。"Exort task data" --> "LabelMe 3.0"  
(2). 使用[cvat2labelme](https://git.utapp.cn/yuanhui/patrol_ai/-/blob/main/python_codes/lib_label_ops.py#L302)函数将.xml标签文件转换成labelme工具可读的.json标签文件。  
(3). 使用[labelme_2_coco](https://git.utapp.cn/yuanhui/patrol_ai/-/blob/main/python_codes/lib_label_ops.py#L302)函数将需要训练的标签整合成一个文件"trainval.json"。  
(4). 将标签上传到训练服务器，并用以下结构存放。  
```
  train/
    annotations/
        trainval.json
    images/
        image_1.jpg
        image_2.jpg
        ...
```
##### 2. 训练模型
(1). 准备detectron2环境。安装教程详见[install](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)，也可使用已安装好环境的docker镜像, 如utdnn/inspection:cuda11.4-conda-cuml-opencv等。  
(2). 训练。在安装了detectron2的环境下使用python脚本训练，官方demo可参考[colab](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=wlqXIXXhW8dA)，简易demo可参考[train.py](https://git.utapp.cn/yuanhui/patrol_ai/-/blob/main/shell/maskrcnn/train.py)。  
(3). 训练完成后会生成checkpoint的log文件和.pth模型文件，可使用tensorboard查看log文件。
##### 3. 推理
参考: [colab demo](https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=0e4vdDIOXyxF), [gitlab demo](https://git.utapp.cn/yuanhui/patrol_ai/-/blob/main/python_codes/lib_inference_mrcnn.py)



