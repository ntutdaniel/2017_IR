import file_control as file_c
import numpy as np
import math

pbglm = '../dataset/BGLM.txt'
pcollection = '../dataset/Collection.txt'
start_index = 0
tk = 2  # Tk

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
p_wk = np.random.randint(1, v_count + 1, size=(v_count, tk))
p_wk_col_sum = p_wk.sum(axis=0) + 0.
p_wk = p_wk / p_wk_col_sum

# matrix (tk by document) document document's count = 18461
dc_count = len(collection)
p_kd = np.random.randint(1, dc_count + 1, size=(tk, dc_count))
p_kd_col_sum = p_kd.sum(axis=0) + 0.
p_kd = p_kd / p_kd_col_sum

'''
    v1 v2 
d1
d2
'''
# matrix (tk | wi, dj)
p_kwd = np.zeros(shape=(tk, dc_count, v_count))

'''
tool function
'''


def isNanAndInf(n):
    if np.isnan(n) or np.isinf(n):
        return True
    else:
        return False


def LogAdd(x, y):
    lzero = -1.0e10  # ~=log(0)
    lsmall = -0.5e10  # log values < LSMALL are set to LZERO
    minLogExp = -1 * math.log(-1 * lzero)
    add = 0

    # if x < y:
    #     temp = x
    #     x = y
    #     y = temp
    #
    # diff = y - x #diff < 0
    # print(x,y,diff,np.isinf(diff),x < lsmall)
    #
    # if diff < minLogExp or np.isinf(diff):
    #     if x < lsmall:
    #         return lzero
    #     else:
    #         return x
    # else:
    #     z = math.exp(diff)
    #     return x + math.log(1.0 + z)
    if x == 0:
        add = math.log(y)
    elif y == 0:
        add = math.log(x)
    elif x >= y:
        add = math.log(x) + math.log(1 + math.exp(math.log(y) - math.log(x)))
    else:
        add = math.log(y) + math.log(1 + math.exp(math.log(x) - math.log(y)))

    return add


# test = LogAdd(10, 10)

def GetPTkWiDj(k, i, j):
    global p_wk, p_kd
    global tk

    num = math.log(p_wk[i][k]) + math.log(p_kd[k][j])
    den = 0
    for k_index in range(0, tk):
        temp = math.log(p_wk[i][k_index]) + math.log(p_kd[k_index][j])
        temp = math.exp(temp)
        den = LogAdd(math.exp(den), temp)
    div = num - den
    return div


def RunE():
    global tk, v_count, dc_count
    global p_kwd
    for k_index in range(0, tk):
        for j in range(0, dc_count):
            for i in range(0, v_count):
                p_kwd[k_index][j][i] = math.exp(GetPTkWiDj(k_index, i, j))
                # Debug
                if isNanAndInf(p_kwd[k_index][j][i]):
                    print('RunE', k_index, i, j, p_kwd[k_index][j][i])


def GetWiTk(k, i):
    global p_wk, p_kd
    global v_count, dc_count

    num = 0
    for j in range(0, dc_count):
        if (i in collection[j]):
            temp = math.log(collection[j][i]) + math.log(p_kwd[k][j][i])
            temp = math.exp(temp)
            num = LogAdd(math.exp(num), temp)
            #print(j, num)

    den = 0
    for i_index in range(0, v_count):
        for j in range(0, dc_count):
            if (i_index in collection[j]):
                temp = math.log(collection[j][i_index]) + math.log(p_kwd[k][j][i_index])
                temp = math.exp(temp)
                den = LogAdd(math.exp(den), temp)
                #print(i_index, j, den)
    div = num - den

    # Debug
    if isNanAndInf(div):
        print('p(wi|tk)', i, k, str(div))
    return div


def GetTkDj(k, j):
    global p_wk, p_kd
    global v_count, dc_count

    num = 0
    den = 0
    for i in range(0, v_count):
        if (i in collection[j]):
            temp = math.log(collection[j][i]) + math.log(p_kwd[k][j][i])
            temp = math.exp(temp)
            num = LogAdd(math.exp(num), temp)
            den += collection[j][i]
    den = math.log(den)
    div = num - den
    # Debug
    if isNanAndInf(div):
        print('p(tk|dj)', k, j, str(div))
    return div


def RunM():
    global tk, v_count, dc_count
    global p_wk, p_kd

    for k in range(0, tk):
        for i in range(0, v_count):
            p_wk[i][k] = math.exp(GetWiTk(k, i))

    for k in range(0, tk):
        for j in range(0, dc_count):
            p_kd[k][j] = math.exp(GetTkDj(k, j))


if __name__ == '__main__':
    po_wk = '../dataset/Output/p_init_wk.txt'
    po_kd = '../dataset/Output/p_init_kd.txt'
    np.savetxt(po_wk, p_wk, delimiter=',')
    np.savetxt(po_kd, p_kd, delimiter=',')

    '''
    EM
    '''
    RunE()
    RunM()

    po_wk = '../dataset/Output/p_plsa_wk.txt'
    po_kd = '../dataset/Output/p_plsa_kd.txt'
    np.savetxt(po_wk, p_wk, delimiter=',')
    np.savetxt(po_kd, p_kd, delimiter=',')
