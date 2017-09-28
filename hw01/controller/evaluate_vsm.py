import ir_file as ir_f

if __name__ == '__main__':
    pa = '../dataset/Output/query'
    a_start_index = 0

    '''
    讀取檔案
    '''
    qs = ir_f.ReadEvaFolder(pa, a_start_index)

    temp_qs = {}
    for fn, ds in qs.items():
        if (fn[8:] == '20002.query'): temp_qs[fn] = ds

    ans = ['VOM19980225.0700.0585', 'VOM19980303.0900.0396',
           'VOM19980303.0900.2128', 'VOM19980304.0700.0737',
           'VOM19980304.0700.1058', 'VOM19980305.0700.0703',
           'VOM19980305.0900.2093', 'VOM19980306.0700.0971',
           'VOM19980311.0700.1487', 'VOM19980326.0700.1783',
           'VOM19980523.0700.0189', 'VOM19980621.0700.0565',
           'VOM19980630.0900.0230']

    temp_qs_count = {}
    for fn, ds in temp_qs.items():
        intersection = list(set(ans).intersection(set(ds)))
        temp_qs_count[fn] = len(intersection)

    for fn, c in sorted(temp_qs_count.items(), key=lambda x: x[1], reverse=True):
        print(fn, c)
