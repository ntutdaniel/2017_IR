# coding: utf-8
import ir_file as ir_f
import math
import os
from time import gmtime, strftime


def calDocumantRank(doc_word_count, folder_word_count_distinct, query_word_count, po, d_tf_c, q_tf_c, d_idf_c, q_idf_c,
                    e):
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

    # tf(i,j)
    for (fn, d) in d_tf.items():
        for (word, count) in d.items():
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

    # idf(i,j)
    log_d_n = {}
    for (word, count) in d_n.items():
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
    #     d_temp = {}
    #     for (word, count) in d.items():
    #         print(fn, word, d_tf[fn][word], log_d_n[word], count)

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


    '''
     ouput
    '''
    for q, ds in sim_q.items():
        # now = strftime("%Y%m%d%H%M%S", gmtime())
        temp_p = po + '/query'
        temp_q = str(d_tf_c) + '_' + str(q_tf_c) + '_' + str(d_idf_c) + '_' + str(q_idf_c) + '_' + q

        if (os.path.exists(os.path.join(temp_p, temp_q))):
            os.remove(os.path.join(temp_p, temp_q))
        f = open(os.path.join(temp_p, temp_q), 'w')
        for (d, score) in sorted(dict(sim_q[q]).items(), key=lambda x: x[1], reverse=True):
            f.writelines(str(d) + ": " + str(score) + '\n')
        f.close()


if __name__ == '__main__':
    pd = '../dataset/Document'
    po = '../dataset/Output'
    pq = '../dataset/Query'
    d_start_index = 3
    q_start_index = 0
    e = 0.5

    doc_word_count, folder_word_count, folder_word_count_distinct = ir_f.ReadFolder(pd, d_start_index)
    # ir_f.ReadFolderDebug(po, doc_word_count, folder_word_count, folder_word_count_distinct)

    # query
    query_word_count, query_folder_word_count, query_folder_word_count_distinct = ir_f.ReadFolder(pq, q_start_index)
    # ir_f.ReadFolderDebug(po, doc_word_count, folder_word_count, folder_word_count_distinct)

    calDocumantRank(doc_word_count, folder_word_count_distinct, query_word_count, po, 1, 1, 1, 1, e)
