import file_control as file_c
import PLSA as plsa
import numpy as np
import math
import os
import sys

pd = '../dataset/Document'
po = '../dataset/Output'
pq = '../dataset/Query'
pbglm = '../dataset/BGLM.txt'
start_index = 0
tk = 3

'''
read file
'''
query_word_count, folder_word_count, folder_word_count_distinct = file_c.ReadFolder(pq, start_index)

start_index = 3
doc_word_count, folder_word_count, folder_word_count_distinct = file_c.ReadFolder(pd, start_index)

# for fn, words in query_word_count.items():
#     for word, count in words.items():
#         print(fn, word, count)

bglm = file_c.ReadBGLMFile(pbglm)

docs_index_list = {}
collection = {}  # 2265 document

for i, fs in enumerate(sorted(doc_word_count.keys())):
    docs_index_list[i] = fs

for i, fn in docs_index_list.items():  # 0-2264
    collection[i] = doc_word_count[fn]

# print(bglm)
'''
read matrix(training model) !!!!!!!!!!!!!!!!!!!!!!!!
'''
f_wk = 'p_plsa_wk.txt'
f_kd = 'p_plsa_kd.txt'
p_wk = []
p_kd = []
if len(sys.argv) < 2:
    path_pwk = '../dataset/Output/fold_in_p_init_wk.txt'
    p_wk = np.loadtxt(path_pwk, delimiter=',')

    path_pkd = '../dataset/Output/fold_in_p_init_kd.txt'
    p_kd = np.loadtxt(path_pkd, delimiter=',')
else:
    train_index = sys.argv[1]
    path_pwk = '../dataset/Output/testing/' + 'training' + str(train_index) + '_' + f_wk
    p_wk = np.loadtxt(path_pwk, delimiter=',')

    path_pkd = '../dataset/Output/testing/' + 'training' + str(train_index) + '_' + f_kd
    p_kd = np.loadtxt(path_pkd, delimiter=',')

'''
alpha & betha
'''
alpha = 0.3
betha = 0.3

'''
likelihood
'''


# for fn, words in query_word_count.items():
#     temp_p = 0
#     for word, count in words.items():

def GetPWiDj(i, j):
    global collection
    d = collection[j]
    t = 0
    for word, count in d.items():
        t += count
    # print(t)

    n = 0
    if (i in collection[j]): n = collection[j][i]

    return float(n) / t


def GetPPWiDj(i, j):
    ans = 0
    a = alpha * GetPWiDj(i, j)

    b = 0
    for k in range(0, tk):
        temp = math.log(p_wk[i][k]) + math.log(p_kd[k][j])
        temp = math.exp(temp)
        # den_test += p_wk[i][k_index] * p_kd[k_index][j]
        b = plsa.LogAdd(math.exp(b), temp)
    b = betha * b

    c = (1 - alpha - betha) * math.exp(bglm[i])
    ans = a + b + c
    return ans


def CalcQueryRank():
    q_ranks = {}
    for fn, words in query_word_count.items():
        d_ranks = {}
        for i in range(0, len(collection)):
            rank = 0
            for word, count in words.items():
                rank += math.log(GetPPWiDj(word, i))
            d_ranks[docs_index_list[i]] = math.exp(rank)  # index to doc name
        q_ranks[fn] = d_ranks
    return q_ranks

if __name__ == '__main__':
    q_ranks = CalcQueryRank()

    #debug
    # for q, ds in q_ranks.items():
    #     for d, r in ds.items():
    #         print(q, d, r)

    temp = []
    # temp.append('Query, RetrievedDocuments')
    temp_p = po + '/rank'
    filename = 'likelihood.txt'
    if (os.path.exists(os.path.join(temp_p, filename))):
        os.remove(os.path.join(temp_p, filename))
    f = open(os.path.join(temp_p, filename), 'w')
    for q, ds in q_ranks.items():
        temp_q_a = q + ',';
        #print(len(sorted(dict(q_ranks[q]).items(), key=lambda x: x[1], reverse=True)))
        for (d, score) in sorted(dict(q_ranks[q]).items(), key=lambda x: x[1], reverse=True):
            temp_q_a += str(d) + ' '
        temp_q_a = temp_q_a.strip()
        temp.append(temp_q_a)

    temp = sorted(temp)
    temp.insert(0, "Query,RetrievedDocuments")
    for a in temp:
        # print(a)
        if temp.index(a) != len(temp) - 1: a += '\n'
        f.writelines(a)
    f.close()

