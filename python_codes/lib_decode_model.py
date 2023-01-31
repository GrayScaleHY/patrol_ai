import random
import io
import time
import torch
import numba
import numpy as np


# 解密
def decode_model(encode_model_name, keys=1,length=5000):
    f = open(encode_model_name, "rb")
    encode_model = f.read()
    f.close()
    random.seed(keys)
    # encode_model = bytearray(encode_model)
    encode_model = np.frombuffer(encode_model, dtype=np.uint8)
    byteskey = np.uint8(random.randint(0, 255))

    # time1 = time.time()
    decode_model_ = decode(encode_model, byteskey)
    # time2 = time.time()
    # print(time2-time1)

    decode_model_ = bytes(decode_model_)

    #  写临时文件
    decode_model_name = encode_model_name[:-4] + 'tmp' + encode_model_name[-4:]
    with open(decode_model_name, "wb") as fp:
        fp.write(decode_model_)

    decode_model_ = io.BytesIO(decode_model_)  #返回io对象
    return decode_model_name, decode_model_


@numba.jit
def decode(encode_model, byteskey):
    tmp = np.empty_like(encode_model)
    for i in range(len(encode_model)):
        tmp[i] = encode_model[i] ^ byteskey
    return tmp


if __name__ == '__main__':
    filename = 'hs_bj_infer_encode.pt'
    str_4 = decode_model('hs_bj_infer_encode.pt')
    # print(type(str_4))
    test = torch.load(str_4)

