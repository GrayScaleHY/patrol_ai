import random
import io
import torch
import numba

# 解密
@numba.jit
def decode_model(encode_model_name, keys=1,length=5000):
    f = open(encode_model_name, "rb")
    encode_model = f.read()
    f.close()
    random.seed(keys)
    encode_model = bytearray(encode_model)
    byteskey = random.randint(0, 255)
    for i in range(len(encode_model)):
        encode_model[i] ^= byteskey
    decode_model_ = bytes(encode_model)
    # decode_model_ = io.BytesIO(decode_model_)  #返回io对象
    return decode_model_

if __name__ == '__main__':
    filename = 'hs_bj_infer_encode.pt'
    str_4 = decode_model('hs_bj_infer_encode.pt')
    print(type(str_4))
    test = torch.load(str_4)

