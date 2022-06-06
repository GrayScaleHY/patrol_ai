extern "C"{

__device__ float min_abs_diff_element_wise(
    const float* ref, const float* tgt, int width, int height, \
    int x, int y, int xo_max, int yo_max)
{   
    float res = 1e6;

    int refind = y * width + x;
    int tgtind;
    float absdiff;

    for (int i = -yo_max; i <= yo_max; i++){
        for (int j = -xo_max; j <= xo_max; j++){
            tgtind = (y + i) * width + x + j;
            absdiff = fabs(ref[refind] - tgt[tgtind]);
            res = min(res, absdiff);
        }
    }
    return res;
}

__global__ void cuda_change_map(
    const float* ref, const float* tgt, float* res, \
    int width, int height, int xo_max, int yo_max)
{
    // loop through ref[ym:height - ym, xm:width - xm]

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int N = (width - 2 * xo_max) * (height - 2 * yo_max);
    int x, y;
    float res1, res2;
 
    for (int i = index; i < N; i += stride){
        x = i % (width - 2 * xo_max) + xo_max;
        y = i / (width - 2 * yo_max) + yo_max;
        res1 = min_abs_diff_element_wise(ref, tgt, width, height, x, y, xo_max, yo_max);
        res2 = min_abs_diff_element_wise(tgt, ref, width, height, x, y, xo_max, yo_max);
        res[i] = max(res1, res2);
    }
}

}