import file_control as file_c
import PLSA as plsa
import numpy as np
import math
import sys

pd = '../dataset/Document'
po = '../dataset/Output'
pq = '../dataset/Query'
pbglm = '../dataset/BGLM.txt'
start_index = 3
tk = 10  # Tk
f_wk = 'p_plsa_wk.txt'
f_kd = 'p_plsa_kd.txt'
p_wk = []
p_kd = []
p_wk_old = []
p_kd_old = []

'''
read file (BGLM.txt & Collection.txt)
'''
doc_word_count, folder_word_count, folder_word_count_distinct = file_c.ReadFolder(pd, start_index)

bglm = file_c.ReadBGLMFile(pbglm)

# debug
'''

'''
docs_index_list = {}
collection = {}  # 2265 document

for i, fs in enumerate(sorted(doc_word_count.keys())):
    docs_index_list[i] = fs

for i, fn in docs_index_list.items():  # 0-2264
    collection[i] = doc_word_count[fn]

'''
init 2 matrix
'''
# matrix (word by tk) word's count = 51253
v_count = len(bglm)

dc_count = len(doc_word_count)
p_kd = np.random.randint(1, dc_count + 1, size=(tk, dc_count))
p_kd_col_sum = p_kd.sum(axis=0) + 0.
p_kd = p_kd / p_kd_col_sum

# matrix (tk | wi, dj)
# p_kwd = np.zeros(shape=(tk, dc_count, v_count))

'''
training p(wi|tk) !!!!!!!!!!!!!!!!!!!!!
'''

if len(sys.argv) < 2:
    path_pwk = '../dataset/Output/p_init_wk.txt'
    p_wk = np.loadtxt(path_pwk, delimiter=',')
else:
    train_index = sys.argv[1]
    path_pwk = '../dataset/Output/training/' + 'training' + str(train_index) + '_' + f_wk
    p_wk = np.loadtxt(path_pwk, delimiter=',')

'''
E Step
'''


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
        den = plsa.LogAdd(math.exp(den), temp)
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
        den = plsa.LogAdd(math.exp(den), temp)
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
                # print(p_kwd[k_index][j][i])
                # Debug
                if plsa.isNanAndInf(p_kwd[k_index][j][i]):
                    print('RunE', k_index, i, j, p_kwd[k_index][j][i])


'''
M Step
'''


def GetTkDj(k, j):
    global p_wk, p_kd
    global v_count, dc_count

    num = 0
    den = 0
    # num_test = 0
    for i in range(0, v_count):
        if (i in collection[j]):
            if GetPTkWiDjV2(k, i, j) == 0:
                continue

            temp = math.log(collection[j][i]) + math.log(GetPTkWiDjV2(k, i, j))
            temp = math.exp(temp)
            num = plsa.LogAdd(math.exp(num), temp)
            # num_test += collection[j][i] * p_kwd[k][j][i]
            # print(collection[j][i], p_kwd[k][j][i], math.log(num_test), num)
            den += collection[j][i]

    den = math.log(den)
    div = num - den
    # Debug
    if plsa.isNanAndInf(div):
        print('p(tk|dj)', k, j, str(div))
    # print(div)
    return div


def RunM():
    global tk, dc_count
    global p_kd

    for k in range(0, tk):
        for j in range(0, dc_count):
            p_kd[k][j] = math.exp(GetTkDj(k, j))


if __name__ == '__main__':
    po_wk = '../dataset/Output/fold_in_p_init_wk.txt'
    po_kd = '../dataset/Output/fold_in_p_init_kd.txt'
    np.savetxt(po_wk, p_wk, delimiter=',')
    np.savetxt(po_kd, p_kd, delimiter=',')

    train_index = 0
    train_total = 10

    while train_index < train_total:
        p_kd_old = p_kd
        p_wk_old = p_wk
        # print(p_kd)
        # print(p_wk)
        #print(len(p_kd_old), len(p_kd_old[0]))
        #print(len(p_wk_old), len(p_wk_old[0]))
        '''
        EM
        '''
        # print('testing E processing...' + str(train_index) + '/' + str(train_total - 1))
        # RunE()
        print('testing M processing...' + str(train_index) + '/' + str(train_total - 1))
        RunM()

        p_wk = plsa.probNorm(p_wk)
        p_kd = plsa.probNorm(p_kd)

        f_wk = 'p_plsa_wk.txt'
        f_kd = 'p_plsa_kd.txt'
        po_wk = '../dataset/Output/testing/testing' + str(train_index) + '_' + f_wk
        po_kd = '../dataset/Output/testing/testing' + str(train_index) + '_' + f_kd
        np.savetxt(po_wk, p_wk, delimiter=',')
        np.savetxt(po_kd, p_kd, delimiter=',')

        train_index += 1
