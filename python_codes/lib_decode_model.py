import random
import io
import time
import torch
import numba
import numpy as np
from xor import xor  # cython

# 解密
def decode_model(encode_model_name, keys=1, length=5000):
    f = open(encode_model_name, "rb")
    encode_model_ = f.read()
    f.close()
    random.seed(keys)
    encode_model_ = np.frombuffer(bytearray(encode_model_), dtype=np.uint8)
    byteskey = np.uint8(random.randint(0, 255))

    time1 = time.time()
    decode_model_ = xor(encode_model_, byteskey)
    time2 = time.time()
    print(time2 - time1)

    decode_model_ = bytes(decode_model_)

    #  写临时文件
    decode_model_name = encode_model_name[:-4] + 'tmp' + encode_model_name[-4:]
    with open(decode_model_name, "wb") as fp:
        fp.write(decode_model_)

    decode_model_ = io.BytesIO(decode_model_)  # 返回io对象
    return decode_model_name, decode_model_


"""
@numba.jit
def xor(encode_model_, byteskey):
    tmp = np.empty_like(encode_model_)
    for i in range(len(encode_model_)):
        tmp[i] = encode_model_[i] ^ byteskey
    return tmp
"""


# 加密
def encode_model(origin_model_name, encode_model_name, keys=1, length=5000):
    f = open(origin_model_name, "rb")
    origin_model = f.read()
    f.close()
    random.seed(keys)
    origin_model = np.frombuffer(bytearray(origin_model), dtype=np.uint8)
    byteskey = np.uint8(random.randint(0, 255))

    encode_model_ = xor(origin_model, byteskey)

    encode_model_ = bytes(encode_model_)

    #  写加密文件
    with open(encode_model_name, "wb") as fp:
        fp.write(encode_model_)

    encode_model_ = io.BytesIO(encode_model_)  # 返回io对象
    return encode_model_name, encode_model_


if __name__ == '__main__':
    filename = 'hs_bj_infer_encode.pt'
    str_4 = decode_model('hs_bj_infer_encode.pt')[1]
    print(type(str_4))
    # test = torch.load(str_4)
