import os


def ReadFile(path, start_index):
    f = open(path, 'r')
    rows = f.readlines()

    q_dic = {}
    for (i, row) in enumerate(rows):
        if (i >= start_index):
            temp = row.split(',')
            q = temp[0]
            ans = temp[1].split()
            ans = [{aa.strip(): 0} for aa in ans]
            q_dic[q] = ans

    return q_dic


if __name__ == '__main__':
    debug1 = True
    debug2 = False
    path = '../data/realAns/AssessmentTrainSet.txt'
    start_index = 1  # !!
    q_dic = ReadFile(path, start_index)
    if debug1:
        for q, ds in q_dic.items():
            print(len(ds))
            for d in ds:
                print(q, d)

    path = '../data/solution.txt'
    q_dic = ReadFile(path, start_index)
    if debug2: print(len(q_dic), q_dic)
