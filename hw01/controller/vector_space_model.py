# coding: utf-8
import ir_file as ir_f
import math
import os
from time import gmtime, strftime


def calDocumantRank(pd, pq, po, d_start_index, q_start_index):
    # documents
    doc_word_count, folder_word_count, folder_word_count_distinct = ir_f.ReadFolder(pd, d_start_index)
    # ir_f.ReadFolderDebug(po, doc_word_count, folder_word_count, folder_word_count_distinct)

    # query
    query_word_count, query_folder_word_count, query_folder_word_count_distinct = ir_f.ReadFolder(pq, q_start_index)
    # ir_f.ReadFolderDebug(po, doc_word_count, folder_word_count, folder_word_count_distinct)

    N = len(doc_word_count)
    d_n = folder_word_count_distinct
    d_tf = doc_word_count

    q_tf = query_word_count

    '''
    scheme 01 Document Term Weight
    '''

    # log(N/d_n)
    log_d_n = {}
    for (word, count) in d_n.items():
        log_d_n[word] = math.log(N / count, 10)

    # print(d_n)
    # print(len(log_d_n))


    # tf(i,f) documents
    d_tf_w = {}
    for (fn, d) in d_tf.items():
        d_temp = {}
        for (word, count) in d.items():
            d_temp[word] = (1 + math.log(count, 2)) * log_d_n[word]
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
        q_max = max(list(q.values()))
        for (word, count) in q.items():
            idx = 0
            if (word not in log_d_n):
                idx = 0  # !!
            else:
                idx = log_d_n[word]
            q_temp[word] = (0.5 + 0.5 * (count / q_max)) * idx
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
        #now = strftime("%Y%m%d%H%M%S", gmtime())
        temp_p = po + '/query'
        temp_q = q

        if (os.path.exists(os.path.join(temp_p, temp_q))):
            os.remove(os.path.join(temp_p, temp_q))
        f = open(os.path.join(temp_p, temp_q), 'xt')
        for (d, score) in sorted(dict(sim_q[q]).items(), key=lambda x: x[1], reverse=True):
            f.writelines(str(d) + ": " + str(score) + '\n')
        f.close()


if __name__ == '__main__':
    pd = '../dataset/Document'
    po = '../dataset/Output'
    pq = '../dataset/Query'
    d_start_index = 3
    q_start_index = 0
    calDocumantRank(pd, pq, po, d_start_index, q_start_index)
