# coding: utf-8
import ir_file as ir_f
import math
import os
import pyodbc
from time import gmtime, strftime
import numpy as np


def calDocumantRank(doc_word_count, folder_word_count_distinct, query_word_count, po, d_tf_c, q_tf_c, d_idf_c, q_idf_c,
                    e, qr_c):
    # documents
    # doc_word_count, folder_word_count, folder_word_count_distinct = ir_f.ReadFolder(pd, d_start_index)
    # ir_f.ReadFolderDebug(po, doc_word_count, folder_word_count, folder_word_count_distinct)

    # query
    # query_word_count, query_folder_word_count, query_folder_word_count_distinct = ir_f.ReadFolder(pq, q_start_index)
    # ir_f.ReadFolderDebug(po, doc_word_count, folder_word_count, folder_word_count_distinct)

    N = len(doc_word_count)
    d_n = folder_word_count_distinct
    d_tf = doc_word_count

    q_tf = query_word_count

    d_len = {}
    # each doc length
    for (fn, d) in d_tf.items():
        wc = 0
        for (word, count) in d.items():
            wc += count
        d_len[fn] = wc

    d_avg_len = sum(list(d_len.values())) / float(len(d_len))

    b = 0.9
    tfp = {}
    for (fn, d) in d_tf.items():
        temp = {}
        for (word, count) in d.items():
            temp[word] = count / ((1 - b) + (b * len(d)) / d_avg_len)
        tfp[fn] = temp

    k1 = 10.0
    k3 = 3.0

    # tf(i,j)
    for (fn, d) in d_tf.items():
        for (word, count) in d.items():
            # print(fn, word, (tfp[fn])[word])
            if (d_tf_c == 1):
                if (d[word] > 0):
                    d[word] = 1
                else:
                    d[word] = 0
            elif (d_tf_c == 2):  # !!
                d[word] = count
            elif (d_tf_c == 3):  # !!
                d[word] = 1 + math.log(count, 2)
            elif (d_tf_c == 4):
                d_max = max(list(d.values()))
                d[word] = (0.5 + 0.5 * float(count / d_max))
            elif (d_tf_c == 5):
                d_max = max(list(d.values()))
                d[word] = (e + (1 - e) * float(count / d_max))
            elif (d_tf_c == 6):  # sm25
                d[word] = (k1 + 1) * (tfp[fn])[word] / (k1 + (tfp[fn])[word])

    # for (fn, d) in d_tf.items():
    #     for (word, count) in d.items():
    #         print(fn, word, count)

    # tf(i,q)
    for (fn, d) in q_tf.items():
        for (word, count) in d.items():
            if (q_tf_c == 1):
                if (d[word] > 0):
                    d[word] = 1
                else:
                    d[word] = 0
            elif (q_tf_c == 2):  # !!
                d[word] = count
            elif (q_tf_c == 3):  # !!
                d[word] = 1 + math.log(count, 2)
            elif (q_tf_c == 4):
                d_max = max(list(d.values()))
                d[word] = (0.5 + 0.5 * float(count / d_max))
            elif (q_tf_c == 5):
                d_max = max(list(d.values()))
                d[word] = (e + (1 - e) * float(count / d_max))
            elif (q_tf_c == 6):
                d[word] = (k3 + 1) * count / (k3 + count)

    # idf(i,j)
    log_d_n = {}
    for (word, count) in d_n.items():
        # print(word, count)
        if d_idf_c == 1:
            log_d_n[word] = 1
        elif d_idf_c == 2:
            log_d_n[word] = math.log(float(N / count), 10)
        elif d_idf_c == 3:
            log_d_n[word] = math.log(1 + float(N / count), 10)
        elif d_idf_c == 4:
            q_max = max(list(d_n.values()))
            log_d_n[word] = math.log(1 + float(q_max / count), 10)
        elif d_idf_c == 5:
            log_d_n[word] = math.log(float(N - count) / count, 10)
        elif d_idf_c == 6:
            log_d_n[word] = math.log(float(N - count + 0.5) / count + 0.5, 10)

    # for (word, count) in log_d_n.items():
    #     print(word, count)

    # idf(i,q)
    log_q_n = {}  # !!equal to log_d_n
    for (word, count) in d_n.items():
        if q_idf_c == 1:
            log_q_n[word] = 1
        elif q_idf_c == 2:
            log_q_n[word] = math.log(float(N / count), 10)
        elif q_idf_c == 3:
            log_q_n[word] = math.log(1 + float(N / count), 10)
        elif q_idf_c == 4:
            if count == 0:
                log_q_n[word] = 0
            else:
                q_max = max(list(d_n.values()))
                log_q_n[word] = math.log(1 + float(q_max / count), 10)
        elif q_idf_c == 5:
            if count == 0:
                log_q_n[word] = 0
            else:
                log_q_n[word] = math.log(float(N - count) / count, 10)

    '''
    scheme 01 Document Term Weight
    '''
    # documents term weight
    d_tf_w = {}
    for (fn, d) in d_tf.items():
        d_temp = {}
        for (word, count) in d.items():
            d_temp[word] = count * log_d_n[word]
        d_tf_w[fn] = d_temp

    # for(fn, d) in d_tf_w.items():
    #     for (word, count) in d.items():
    #         print(fn, word, count)

    '''
    scheme 01 Query Term Weight
    '''
    q_tf_w = {}
    for (fn, q) in q_tf.items():
        q_temp = {}
        for (word, count) in q.items():
            idf = 0
            if (word not in log_q_n):
                idf = 0  # !!
            else:
                idf = log_q_n[word]
            q_temp[word] = count * idf
        q_tf_w[fn] = q_temp

    # for(fn, d) in q_tf_w.items():
    #     for (word, count) in d.items():
    #         print(fn, word, count)

    '''
     scheme 01 Document Term Weight x Query Term Weight
    '''
    # read embeding
    path_pwk = '../dataset/embeding_20171118210556814173_42'
    embeding = np.loadtxt(path_pwk, delimiter=',')
    # print(len(embeding),len(embeding[0]))
    # print(embeding[0])
    # print(embeding[0][0])
    # print(type(embeding[0][0]))

    # sim
    dim = 100
    # temp_array = np.random.randint(10, size=(1, dim))
    # temp_array2 = np.random.randint(10, size=(1, dim))
    sim_q = {}
    for (fq, q) in q_tf_w.items():
        q_len = len(q.values())

        # query vector
        q_vector = np.zeros(dim)
        for word, count in q.items():
            q_vector = q_vector + (float(count) / q_len) * embeding[int(word)]  # temp_array
            # print(q_vector)

        # document vec
        sim_d = {}
        for (fd, d) in d_tf_w.items():
            d_vector = np.zeros(dim)
            d_len = len(d.values())
            for word, count in d.items():
                d_vector = d_vector + (float(count) / d_len) * embeding[int(word)]  # temp_array

            # cos doc and query !!index
            cos_num = sum(q_vector * d_vector)
            cos_den = sum(q_vector ** 2) ** (0.5) * sum(d_vector ** 2) ** (0.5)
            cos = 0

            if cos_num == 0:
                cos = 0
            elif cos_den == 0:
                cos = -0.5E10
            else:
                cos = math.log(abs(cos_num)) - math.log(abs(cos_den))
                cos = math.exp(cos)
                if np.sign(cos_num) * np.sign(cos_den) == 1:
                    cos = cos * 1
                else:
                    cos = cos * -1

            sim_d[fd] = cos

        sorted_sim_d = sorted(sim_d.items(), key=lambda x: x[1], reverse=True)
        sim_q[fq] = sorted_sim_d
        # print(fq)
        # print(sim_d)
        # print(sorted_sim_d)
        # print('----')
    # print(dict(sim_q['20002.query']))
    # print(sorted(dict(sim_q['20002.query']).items(), key=lambda x: x[1], reverse=True))


    is_hw01 = True
    '''
     ouput origin(document, ranking)
    '''
    if (is_hw01 == False):
        for q, ds in sim_q.items():
            # now = strftime("%Y%m%d%H%M%S", gmtime())
            temp_p = po + '/query'
            temp_q = str(d_tf_c) + '_' + str(q_tf_c) + '_' + str(d_idf_c) + '_' + str(q_idf_c) + '_' + q

            if (os.path.exists(os.path.join(temp_p, temp_q))):
                os.remove(os.path.join(temp_p, temp_q))
            f = open(os.path.join(temp_p, temp_q), 'w')
            for (d, score) in sorted(dict(sim_q[q]).items(), key=lambda x: x[1], reverse=True)[:qr_c]:
                f.writelines(str(d) + ": " + str(score) + '\n')
            f.close()
    else:
        temp = []
        # temp.append('Query, RetrievedDocuments')
        temp_p = po + '/Q_RD'
        filename = str(d_tf_c) + '_' + str(q_tf_c) + '_' + str(d_idf_c) + '_' + str(q_idf_c) + '_' + 'hw01_answer'
        if (os.path.exists(os.path.join(temp_p, filename))):
            os.remove(os.path.join(temp_p, filename))
        f = open(os.path.join(temp_p, filename), 'w')
        for q, ds in sim_q.items():
            temp_q_a = q + ',';
            for (d, score) in sorted(dict(sim_q[q]).items(), key=lambda x: x[1], reverse=True)[:qr_c]:
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


if __name__ == '__main__':
    pd = '../dataset/Document_hw5'
    po = '../dataset/Output'
    pq = '../dataset/Query_hw5'
    d_start_index = 3
    q_start_index = 0
    e = 0.5
    qr_c = 2265

    doc_word_count, folder_word_count, folder_word_count_distinct = ir_f.ReadFolder(pd, d_start_index)
    # ir_f.ReadFolderDebug(po, doc_word_count, folder_word_count, folder_word_count_distinct)

    # query
    query_word_count, query_folder_word_count, query_folder_word_count_distinct = ir_f.ReadFolder(pq, q_start_index)
    # ir_f.ReadFolderDebug(po, doc_word_count, folder_word_count, folder_word_count_distinct)

    calDocumantRank(doc_word_count, folder_word_count_distinct, query_word_count, po, 2, 3, 1, 4, e, qr_c)
