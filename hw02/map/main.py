import os
import map as map

if __name__ == '__main__':
    print('processing...')
    p = '../data/realAns/Q_RD'
    pa = '../data/realAns/AssessmentTrainSet.txt'
    po = '../data/realAns/maps.txt'
    start_index = 1
    files = os.listdir(p)

    maps = {}
    for i, f in enumerate(files):
        fullpath = os.path.join(p, f)
        # print(fullpath)
        base = os.path.basename(fullpath)
        base = os.path.splitext(base)[0]

        print(str(i + 1) + '/' + str(len(files)))
        maps[base] = map.mapFun(fullpath, pa, start_index)

    # 寫黨
    if (os.path.exists(po)):
        os.remove(po)
    f = open(po, 'w')
    for k, v in sorted(dict(maps).items(), key=lambda x: x[1], reverse=True):
        s = str(k) + ': ' + str(v) + '\n'
        f.writelines(s)
    f.close()
    print('done')
