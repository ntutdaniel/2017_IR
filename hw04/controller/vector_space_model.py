# coding: utf-8
import ir_file as ir_f
import math
import os
import copy
#import pyodbc
from time import gmtime, strftime


def calDocumantRank(doc_word_count, folder_word_count_distinct, query_word_count, po, d_tf_c, q_tf_c, d_idf_c, q_idf_c,
                    e, qr_c):
    # documents
    # doc_word_count, folder_word_count, folder_word_count_distinct = ir_f.ReadFolder(pd, d_start_index)
    # ir_f.ReadFolderDebug(po, doc_word_count, folder_word_count, folder_word_count_distinct)

    # query
    # query_word_count, query_folder_word_count, query_folder_word_count_distinct = ir_f.ReadFolder(pq, q_start_index)
    # ir_f.ReadFolderDebug(po, doc_word_count, folder_word_count, folder_word_count_distinct)

    N = len(doc_word_count)
    d_n = folder_word_count_distinct.copy()
    d_tf = doc_word_count.copy()

    q_tf = query_word_count.copy()

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
        #print(word, count)
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
    sim_q = {}
    for (fq, q) in q_tf_w.items():
        keys = list(q.keys())
        values = list(q.values())

        sim_d = {}
        sorted_sim_d = {}

        sum1 = 0
        for v in values:
            sum1 += (v ** 2)

        for (fd, d) in d_tf_w.items():
            qw = {}
            dw = {}
            sum0 = 0
            sum2 = 0
            count = 0
            for key in keys:
                if (key in d):
                    sum0 += d[key] * q[key]
                    # sum1 += d[key] ** 2
                    # sum2 += q[key] ** 2
                    dw[key] = d[key]
                    qw[key] = q[key]
                    count += 1

            for word, count in d.items():
                sum2 += count ** 2

            if (sum1 ** 0.5 * sum2 ** 0.5) == 0:
                sim = 0
            else:
                if d_idf_c == 6 and q_idf_c == 6:
                    sim = sum0  # sm25
                else:
                    sim = sum0 / (sum1 ** 0.5 * sum2 ** 0.5)
            sim_d[fd] = sim
        sorted_sim_d = sorted(sim_d.items(), key=lambda x: x[1], reverse=True)
        sim_q[fq] = sorted_sim_d
        # print(fq)
        # print(sim_d)
        # print(sorted_sim_d)
        # print('----')
    # print(dict(sim_q['20002.query']))
    # print(sorted(dict(sim_q['20002.query']).items(), key=lambda x: x[1], reverse=True))
    return sim_q

def outputfile(sim_q,is_hw01,po,d_tf_c, q_tf_c, d_idf_c, q_idf_c, qr_c):
    '''
     ouput origin(document, ranking)
    '''
    if (is_hw01 == False):
        for q, ds in sim_q.items():
            # now = strftime("%Y%m%d%H%M%S", gmtime())
            temp_p = po + '/Query'
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

def Rocchio(sim_q, query_word_count,doc_word_count,R,A,B):
    q_pl={}
    for q, docs in sim_q.items():
        dc_temp = docs[:R]
        q_temp = query_word_count[q].copy()
        #print q,q_temp

        # cal query weight
        for w,c in q_temp.items():
            q_temp[w] = c * A
            #print q,w,c,q_temp[w]

        d_temp = {}
        # cal document weight
        B_div_R = float(B) / R
        for (doc, rank) in dc_temp:
            d_temp[doc] = doc_word_count[doc].copy()
            for w, c in d_temp[doc].items():
                d_temp[doc][w] = c * B_div_R
            #print q,doc

        #combine
        for d, d_words in d_temp.items():
            for d_w,d_c in d_words.items():
                if d_w in q_temp:
                    q_temp[d_w] = q_temp[d_w] + d_c
                else:
                    q_temp[d_w] = d_c

        q_pl[q] = q_temp
    return q_pl










if __name__ == '__main__':
    pd = '../dataset/Document'
    po = '../dataset/Output'
    pq = '../dataset/Query'
    d_start_index = 3
    q_start_index = 0
    e = 0.5
    qr_c = 100
    is_hw01 = True

    R_c = 10
    A = 0.6
    B = 0.4

    doc_word_count, folder_word_count, folder_word_count_distinct = ir_f.ReadFolder(pd, d_start_index)
    # ir_f.ReadFolderDebug(po, doc_word_count, folder_word_count, folder_word_count_distinct)

    # query
    query_word_count, query_folder_word_count, query_folder_word_count_distinct = ir_f.ReadFolder(pq, q_start_index)
    # for q,words in doc_word_count.items():
    #     print q,words
    q_w_c = copy.deepcopy(query_word_count)
    q_w_c1 = copy.deepcopy(query_word_count)
    #q_w_c2 = copy.deepcopy(query_word_count)

    d_w_c = copy.deepcopy(doc_word_count)
    d_w_c1 = copy.deepcopy(doc_word_count)
    d_w_c2 = copy.deepcopy(doc_word_count)

    d_w_c_d = copy.deepcopy(folder_word_count_distinct)
    #d_w_c_d1 = copy.deepcopy(folder_word_count_distinct)
    d_w_c_d2 = copy.deepcopy(folder_word_count_distinct)

    sim_q = calDocumantRank(d_w_c, d_w_c_d, q_w_c, po, 2, 3, 1, 4, e, qr_c)
    q_ro = Rocchio(sim_q,q_w_c1,d_w_c1,R_c,A,B)
    sim_q = calDocumantRank(d_w_c2, d_w_c_d2, q_ro, po, 2, 3, 1, 4, e, qr_c)
    outputfile(sim_q, is_hw01, po, 2, 3, 1, 4, qr_c)

