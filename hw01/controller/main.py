from pip._vendor.distlib.compat import raw_input
import vector_space_model as vsm

if __name__ == '__main__':
    print('1. Choose tf(i,j) method')
    print('(1.) tf(i,j)')
    print('(2.) 1 + log2(tf(i,j))')
    tf_c = int(raw_input('tf_c(1-2): '))
    while(tf_c > 2 or tf_c < 1) :tf_c = int(raw_input('tf_c(1-2): '))

    print('1. Choose Document Term weight method')
    print('(1.) tf(i,j) * log(N/ni)')
    print('(2.) 1 + tf(i,j)')
    print('(3.) (1 + tf(i,j)) * log(N/ni)')
    dtw = int(raw_input('DTW(1-3): '))
    while (dtw > 3 or dtw < 1): dtw = int(raw_input('DTW(1-3): '))

    print('Choose Query Term weight method')
    print('(1.) (0.5 + 0.5 * (tf(i,q)/maxi(tf(i,q)))) * log(N/ni)')
    print('(2.) log(1 + N/ni)')
    print('(3.) (1 + tf(i,q)) * log(N/ni)')
    qtw = int(raw_input('QTW(1-3): '))
    while (qtw > 3 or qtw < 1): qtw = int(raw_input('QTW(1-3): '))

    print('processing...')

    pd = '../dataset/Document'
    po = '../dataset/Output'
    pq = '../dataset/Query'
    d_start_index = 3
    q_start_index = 0
    vsm.calDocumantRank(pd, pq, po, d_start_index, q_start_index, tf_c, dtw, qtw)