from pip._vendor.distlib.compat import raw_input

if __name__ == '__main__':
    print('Choose Document Term weight method')
    print('1: tf(i,j) * log(N/ni)')
    print('2: 1 + tf(i,j)')
    print('1: (1 + tf(i,j)) * log(N/ni)')
    dtw = int(raw_input('DTW: '))

    print('Choose Query Term weight method')
    print('1: (0.5 + 0.5 * (tf(i,q)/maxi(tf(i,q)))) * log(N/ni)')
    print('2: log(1 + N/ni)')
    print('1: (1 + tf(i,q)) * log(N/ni)')
    qtw = int(raw_input('QTW: '))
