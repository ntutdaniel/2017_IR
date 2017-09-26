from pip._vendor.distlib.compat import raw_input
import vector_space_model as vsm

if __name__ == '__main__':
    pd = '../dataset/Document'
    po = '../dataset/Output'
    pq = '../dataset/Query'
    d_start_index = 3
    q_start_index = 0
    e = 0.5

    print('If run all recommended TF-IDF weighting schemes(y/n)')
    all_c = raw_input('(y/n): ')

    if (all_c == 'n'):
        # document tf
        print('1. Choose tf(i,j) method')
        print('(1.) {0, 1}')
        print('(2.) tf(i,j)')
        print('(3.) 1 + log2(tf(i,j))')
        print('(4.) (0.5 + 0.5 * (tf(i,j)/maxj(tf(i,j))))')
        print('(5.) (e + (1 - e) * (tf(i,j)/maxj(tf(i,j))))')
        d_tf_c = int(raw_input('d_tf_c(1-5): '))
        while (d_tf_c > 5 or d_tf_c < 1): d_tf_c = int(raw_input('try d_tf_c(1-5): '))

        #query tf
        print('2. Choose tf(i,q) method')
        print('(1.) {0, 1}')
        print('(2.) tf(i,q)')
        print('(3.) 1 + log2(tf(i,q))')
        print('(4.) (0.5 + 0.5 * (tf(i,q)/maxq(tf(i,q))))')
        print('(5.) (e + (1 - e) * (tf(i,q)/maxq(tf(i,q))))')
        q_tf_c = int(raw_input('q_tf_c(1-5): '))
        while (q_tf_c > 5 or q_tf_c < 1): q_tf_c = int(raw_input('try q_tf_c(1-5): '))

        if (d_tf_c == 5 or q_tf_c == 5):
            print('Input parameter e')
            e = int(raw_input("d_tf_c's e(0-1): "))
            while (e > 1 or e < 0): e = int(raw_input("e(0-1): "))

        #document idf
        print('3. Choose idf(i,j) method')
        print('(1.) 1')
        print('(2.) log(N/ni)')
        print('(3.) log(1 + N/ni)')
        print('(4.) log(1 + maxi(ni)/ni)')
        print('(5.) log((N - ni)/ni)')
        d_idf_c = int(raw_input('d_idf_c(1-5): '))
        while (d_idf_c > 3 or d_idf_c < 1): d_idf_c = int(raw_input('try d_idf_c(1-5): '))

        # document idf
        print('4. Choose idf(i,q) method')
        print('(1.) 1')
        print('(2.) log(N/ni)')
        print('(3.) log(1 + N/ni)')
        print('(4.) log(1 + maxi(ni)/ni)')
        print('(5.) log((N - ni)/ni)')
        q_idf_c = int(raw_input('q_idf_c(1-5): '))
        while (q_idf_c > 3 or q_idf_c < 1): q_idf_c = int(raw_input('try q_idf_c(1-5): '))

        print('processing...')

        vsm.calDocumantRank(pd, pq, po, d_start_index, q_start_index, d_tf_c, q_tf_c, d_idf_c, q_idf_c, e)
    else:
        for i in range(1, 6, 1):
            for j in range(1,6,1):
                for k in range(1,6,1):
                    for l in range(1, 6, 1):
                        vsm.calDocumantRank(pd, pq, po, d_start_index, q_start_index, i, j, k, l, e)

    print ('done!')
