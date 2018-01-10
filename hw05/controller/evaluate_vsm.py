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
        if (fn[8:] == '20002.query'):
            temp_qs[fn] = ds
        if (fn[8:] == '20001.query'):
            temp_qs[fn] = ds
        if (fn[8:] == '20005.query'):
            temp_qs[fn] = ds
        if (fn[8:] == '20013.query'):
            temp_qs[fn] = ds

    # 10
    ans20002 = [
        'VOM19980225.0700.0585',
        'VOM19980303.0900.0396',
        'VOM19980303.0900.2128',
        'VOM19980304.0700.0737',
        'VOM19980304.0700.1058',
        'VOM19980305.0700.0703',
        'VOM19980305.0900.2093',
        'VOM19980306.0700.0971',
        'VOM19980311.0700.1487',
        'VOM19980326.0700.1793',
        'VOM19980523.0700.0189',
        'VOM19980621.0700.0565',
        'VOM19980630.0900.0230']

    # 13
    ans20001 = [
        'VOM19980220.0700.1159',
        'VOM19980220.0700.2208',
        'VOM19980220.0900.3504',
        'VOM19980221.0700.2087',
        'VOM19980221.0700.2690',
        'VOM19980222.0700.0448',
        'VOM19980223.0700.0562',
        'VOM19980224.0700.0330',
        'VOM19980224.0700.0588',
        'VOM19980225.0900.2949',
        'VOM19980226.0900.3196',
        'VOM19980227.0900.2729',
        'VOM19980228.0700.0268']
    # 10
    ans20005 = [
        'VOM19980224.0900.2290',
        'VOM19980507.0700.0521',
        'VOM19980511.0700.0039',
        'VOM19980511.0700.0080',
        'VOM19980511.0730.0003',
        'VOM19980511.0730.0040',
        'VOM19980511.0900.0004',
        'VOM19980529.0700.0334',
        'VOM19980529.0730.0148',
        'VOM19980529.0900.0148'
    ]

    ans20013 = [
        'VOM19980220.0700.0707',
        'VOM19980220.0700.2814',
        'VOM19980220.0900.3559',
        'VOM19980221.0700.3085',
        'VOM19980223.0700.1511',
    ]


    temp_qs_count = {}
    for fn, ds in temp_qs.items():
        if (fn[8:] == '20002.query'):
            intersection = list(set(ans20002).intersection(set(ds)))
            temp_qs_count[fn] = len(intersection)
        if (fn[8:] == '20001.query'):
            intersection = list(set(ans20001).intersection(set(ds)))
            temp_qs_count[fn] = len(intersection)
        if (fn[8:] == '20005.query'):
            intersection = list(set(ans20005).intersection(set(ds)))
            temp_qs_count[fn] = len(intersection)
        if (fn[8:] == '20013.query'):
            intersection = list(set(ans20013).intersection(set(ds)))
            temp_qs_count[fn] = len(intersection)

    for fn, c in sorted(temp_qs_count.items(), key=lambda x: x[1], reverse=True):
        print(fn, c)
