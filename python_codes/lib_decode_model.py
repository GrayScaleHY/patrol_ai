import random
import io
import time
import numba
import numpy as np
# from xor import xor  # cython

# 解密
def decode_model(encode_model_name, keys=1):
    decode_model_ = xor_model(encode_model_name,keys)

    #  写临时文件
    decode_model_name = encode_model_name[:-4] + 'tmp' + encode_model_name[-4:]
    with open(decode_model_name, "wb") as fp:
        fp.write(decode_model_)

    decode_model_ = io.BytesIO(decode_model_)  # 返回io对象
    return decode_model_name, decode_model_


# @numba.jit
def xor(encode_model_, byteskey):
    tmp = np.empty_like(encode_model_)
    for i in range(len(encode_model_)):
        tmp[i] = encode_model_[i] ^ byteskey
    return tmp


def xor_model(model_name,keys=1):
    f = open(model_name, "rb")
    origin_model = f.read()
    f.close()
    random.seed(keys)
    origin_model = np.frombuffer(bytearray(origin_model), dtype=np.uint8)
    mykey = np.uint8(random.randint(0, 255))

    new_model = xor(origin_model, mykey)
    new_model = bytes(new_model)

    return new_model


# 加密
def encode_model(origin_model_name, encode_model_name, keys=1):
    encode_model_ = xor_model(origin_model_name,keys)

    #  写加密文件
    with open(encode_model_name, "wb") as fp:
        fp.write(encode_model_)

    encode_model_ = io.BytesIO(encode_model_)  # 返回io对象
    return encode_model_name, encode_model_


if __name__ == '__main__':
    filename = r'C:\Users\linduaner.UT\Desktop\task_shanxi\training1.zip'
    str_4 = decode_model(filename)[1]
    print(type(str_4))
    # test = torch.load(str_4)
