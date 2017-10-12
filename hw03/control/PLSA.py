import file_control as file_c
import numpy as np
import math

pbglm = '../dataset/BGLM.txt'
pcollection = '../dataset/Collection.txt'
start_index = 0
k = 2  # Tk

'''
read file (BGLM.txt & Collection.txt)
'''
# BGLM type(dict{world, p(wi)})
bglm = file_c.ReadBGLMFile(pbglm)
# collection(train data) type(dict(collection, dict(word, count)))
collection = file_c.ReadCollectionFile(pcollection)

'''
init 2 matrix
'''
# matrix (word by tk) word's count = 51253
v_count = len(bglm)
p_wk = np.random.randint(1, v_count + 1, size=(v_count, k))
p_wk_col_sum = p_wk.sum(axis=0) + 0.
p_wk = p_wk / p_wk_col_sum

# matrix (tk by document) document document's count = 18461
dc_count = len(collection)
p_kd = np.random.randint(1, dc_count + 1, size=(k, dc_count))
p_kd_col_sum = p_kd.sum(axis=0) + 0.
p_kd = p_kd / p_kd_col_sum

'''
    v1 v2 
d1
d2
'''
# matrix (tk | wi, dj)
p_kwd = np.zeros(shape=(k, dc_count, v_count))


def LogAdd(x, y):
    e = math.exp(1)
    print(e)
    lzero = -1.0E10  # ~=log(0)
    lsmall = -0.5E10  # log values < LSMALL are set to LZERO
    minLogExp = -1 * math.log(-1 * lzero, e)

    if x < y:
        temp = x
        x = y
        y = temp

    diff = y - x

    if diff < minLogExp:
        if x < lsmall:
            return lzero
        else:
            return x
    else:
        z = math.exp(diff)
        return x + math.log(1.0 + z, e)

#test = LogAdd(10, 10)
