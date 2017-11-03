import file_control as file_c
import numpy as np
import math
import sys
import time

pbglm = '../dataset/BGLM.txt'
pcollection = '../dataset/Collection.txt'
start_index = 0
tk = 3  # Tk
f_wk = 'p_plsa_wk.txt'
f_kd = 'p_plsa_kd.txt'
p_wk = []
p_kd = []
p_wk_old = []
p_kd_old = []
bglm = {}
collection = {}
v_count = 0
dc_count = 0
train_index = 0


def init():
    global p_wk, p_kd, bglm, collection, v_count, dc_count, train_index
    '''
    read file (BGLM.txt & Collection.txt)
    '''
    # BGLM type(dict{world, p(wi)})
    bglm = file_c.ReadBGLMFile(pbglm)
    # collection(train data) type(dict(collection, dict(word, count)))
    collection = file_c.ReadCollectionFile(pcollection)
    v_count = len(bglm)
    dc_count = len(collection)

    '''
    init 2 matrix
    '''
    # matrix (word by tk) word's count = 51253
    # matrix (tk by document) document document's count = 18461


    if len(sys.argv) < 2:
        p_wk = np.random.randint(1, v_count + 1, size=(v_count, tk))
        p_wk_col_sum = p_wk.sum(axis=0) + 0.
        p_wk = p_wk / p_wk_col_sum

        p_kd = np.random.randint(1, dc_count + 1, size=(tk, dc_count))
        p_kd_col_sum = p_kd.sum(axis=0) + 0.
        p_kd = p_kd / p_kd_col_sum
    else:
        train_index = sys.argv[1]
        path_pwk = '../dataset/Output/training/' + 'training' + str(train_index) + '_' + f_wk
        p_wk = np.loadtxt(path_pwk, delimiter=',')

        path_pkd = '../dataset/Output/training/' + 'training' + str(train_index) + '_' + f_kd
        p_kd = np.loadtxt(path_pkd, delimiter=',')


def probNorm(matrix):
    m_s = matrix.sum(axis=0) + 0.
    return matrix / m_s


'''
    v1 v2 
d1
d2
'''

# matrix (tk | wi, dj)
# p_kwd = np.zeros(shape=(tk, dc_count, v_count))

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
    # den_test = 0
    for k_index in range(0, tk):
        if p_wk[i][k_index] == 0 or p_kd[k_index][j] == 0:
            continue

        temp = math.log(p_wk[i][k_index]) + math.log(p_kd[k_index][j])
        temp = math.exp(temp)
        # den_test += p_wk[i][k_index] * p_kd[k_index][j]
        den = LogAdd(math.exp(den), temp)
    # print(num, math.log(den_test), math.log(den))

    div = num - math.log(den)
    # print(div)
    return div


def GetPTkWiDjV2(k, i, j):
    global p_wk, p_kd
    global tk

    num = math.log(p_wk_old[i][k]) + math.log(p_kd_old[k][j])
    den = 0
    # den_test = 0
    for k_index in range(0, tk):
        if p_wk_old[i][k_index] == 0 or p_kd_old[k_index][j] == 0:
            continue

        # print(i,j,k_index)
        temp = math.log(p_wk_old[i][k_index]) + math.log(p_kd_old[k_index][j])
        temp = math.exp(temp)
        # den_test += p_wk[i][k_index] * p_kd[k_index][j]
        den = LogAdd(math.exp(den), temp)
    # print(num, math.log(den_test), math.log(den))

    div = num - math.log(den)
    # print(div)
    return math.exp(div)


def RunE():
    global tk, v_count, dc_count
    global p_kwd
    for k_index in range(0, tk):
        for j in range(0, dc_count):
            for i in range(0, v_count):
                if p_wk[i][k_index] == 0 or p_kd[k_index][j] == 0:
                    p_kwd[k_index][j][i] = 0
                else:
                    p_kwd[k_index][j][i] = math.exp(GetPTkWiDj(k_index, i, j))
                # print(math.exp(GetPTkWiDj(0, i, j)),math.exp(GetPTkWiDj(1, i, j)))
                # Debug
                if isNanAndInf(p_kwd[k_index][j][i]):
                    print('RunE', k_index, i, j, p_kwd[k_index][j][i])


def GetWiTk(k, i, den_k):
    global p_wk, p_kd
    global v_count, dc_count

    num = 0
    # num_test = 0
    for j in range(0, dc_count):
        if (i in collection[j]):
            if GetPTkWiDjV2(k, i, j) == 0 or collection[j][i] == 0:
                continue

            # temp = math.log(collection[j][i]) + math.log(p_kwd[k][j][i])
            temp = math.log(collection[j][i]) + math.log(GetPTkWiDjV2(k, i, j))
            temp = math.exp(temp)
            # num_test += collection[j][i] * p_kwd[k][j][i]
            num = LogAdd(math.exp(num), temp)
            # print(math.log(num_test),num)
            # print(j, num)

    den = den_k
    # den_test = 0
    # for i_index in range(0, v_count):
    #     for j in range(0, dc_count):
    #         if (i_index in collection[j]):
    #             if p_kwd[k][j][i_index] == 0:
    #                 continue
    #
    #             temp = math.log(collection[j][i_index]) + math.log(p_kwd[k][j][i_index])
    #             temp = math.exp(temp)
    #             # den_test += collection[j][i_index] * p_kwd[k][j][i_index]
    #             den = LogAdd(math.exp(den), temp)
    #             # print(math.log(den_test),den)
    #             # print(i_index, j, den)
    div = num - den
    # print(div)
    # Debug
    if isNanAndInf(div):
        print('p(wi|tk)', i, k, str(div))
    return div


def checkWordInAllD(i):
    for d, ws in collection.items():
        if i in ws: return True
    return False


def GetWiTkDen(k):
    global p_wk, p_kd  # , p_kwd
    global v_count, dc_count

    den = 0
    '''
    origin
    '''
    # den_test = 0
    # for i_index in range(0, v_count):
    #     for j in range(0, dc_count):
    #         if (i_index in collection[j]):
    #             # if p_kwd[k][j][i_index] == 0:
    #             #     continue
    #             if GetPTkWiDjV2(k, i_index, j) == 0:
    #                 continue
    #
    #             # temp = math.log(collection[j][i_index]) + math.log(p_kwd[k][j][i_index])
    #             temp = math.log(collection[j][i_index]) + math.log(GetPTkWiDjV2(k, i_index, j))
    #             temp = math.exp(temp)
    #             # den_test += collection[j][i_index] * p_kwd[k][j][i_index]
    #             den = LogAdd(math.exp(den), temp)
    #             # print(math.log(den_test),den)
    #             # print(i_index, den)

    '''
    test
    '''
    for j in range(0, dc_count):
        # print(collection[j].keys())
        for i_index in collection[j].keys():
            # print(j,i_index)
            if (i_index in collection[j]):
                if GetPTkWiDjV2(k, i_index, j) == 0:
                    continue

                temp = math.log(collection[j][i_index]) + math.log(GetPTkWiDjV2(k, i_index, j))
                temp = math.exp(temp)
                den = LogAdd(math.exp(den), temp)

    # Debug
    if isNanAndInf(den):
        print('GetWiTkDen_fun', k, str(den))
    return den


def GetTkDj(k, j):
    global p_wk, p_kd
    global v_count, dc_count

    num = 0
    den = 0
    '''
    origin
    '''
    # for i in range(0, v_count):
    #     if (i in collection[j]):
    #         if GetPTkWiDjV2(k, i, j) == 0:
    #             continue
    #
    #         temp = math.log(collection[j][i]) + math.log(GetPTkWiDjV2(k, i, j))
    #         temp = math.exp(temp)
    #         num = LogAdd(math.exp(num), temp)
    #         den += collection[j][i]

    '''
    test
    '''
    for i in collection[j].keys():
        if (i in collection[j]):
            if GetPTkWiDjV2(k, i, j) == 0 or collection[j][i] == 0:
                continue

            temp = math.log(collection[j][i]) + math.log(GetPTkWiDjV2(k, i, j))
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
    # print(v_count,dc_count)
    for k in xrange(0, tk):  # tk
        # den_k = GetWiTkDen(k)  # k den
        # print(k)
        # print('step 01 den done!!')
        b_t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        for i in xrange(0, v_count):  # v_count
            if checkWordInAllD(i):
                p_wk[i][k] = math.exp(GetWiTk(k, i, 0))
            else:
                p_wk[i][k] = 0
            #
            #print(k, i)
            # print(p_wk[i][k])
            # print(p_wk)
        p_wk[:, k] = p_wk[:, k] / (p_wk[:, k].sum(axis=0) + 0.)
        a_t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(p_wk[:, k].sum(axis=0), b_t, a_t)
    print('step 01 done!!')

    for k in xrange(0, tk):
        b_t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        for j in xrange(0, dc_count):
            p_kd[k][j] = math.exp(GetTkDj(k, j))
        a_t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(p_kd[:, k].sum(axis=0), b_t, a_t)
    print('step 02 done!!')


if __name__ == '__main__':
    po_wk = '../dataset/Output/p_init_wk.txt'
    po_kd = '../dataset/Output/p_init_kd.txt'
    np.savetxt(po_wk, p_wk, delimiter=',')
    np.savetxt(po_kd, p_kd, delimiter=',')

    init()
    train_index = int(train_index)
    train_total = 200

    # print(int(train_index), train_total)
    while train_index < train_total:
        '''
        EM
        '''
        # print(train_index)
        p_kd_old = p_kd[:]
        p_wk_old = p_wk[:]
        # print(len(p_kd_old),len(p_kd_old[0]))
        # print(len(p_wk_old),len(p_wk_old[0]))
        # print('E processing...' + str(train_index) + '/' + str(train_total - 1))
        # RunE()
        # print(p_wk[:,0])
        # p_wk[:, 0] = p_wk[:, 0] / 2
        # print(p_wk[:, 0])
        print('M processing...' + str(train_index) + '/' + str(train_total - 1))
        RunM()

        p_wk = probNorm(p_wk)
        p_kd = probNorm(p_kd)

        f_wk = 'p_plsa_wk.txt'
        f_kd = 'p_plsa_kd.txt'
        po_wk = '../dataset/Output/training/training' + str(train_index) + '_' + f_wk
        po_kd = '../dataset/Output/training/training' + str(train_index) + '_' + f_kd
        np.savetxt(po_wk, p_wk, delimiter=',')
        np.savetxt(po_kd, p_kd, delimiter=',')

        train_index += 1
