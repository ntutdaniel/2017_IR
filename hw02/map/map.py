import file_control as file_c


def mapFun(submission, solution, start_index):
    sub_dic = file_c.ReadFile(submission, start_index)
    sol_dic = file_c.ReadFile(solution, start_index)

    for (sub_q, sub_ds) in sub_dic.items():
        for (sol_q, sol_ds) in sol_dic.items():
            if (sub_q == sol_q):
                for i, sub_d in enumerate(sub_ds):
                    sub_k, sub_v = list(sub_d.items())[0]
                    for j, sol_d in enumerate(sol_ds):
                        sol_k, sol_v = list(sol_d.items())[0]
                        if (sub_k == sol_k):
                            sub_ds[i][sub_k] = 1

    q_map = []
    for (sub_q, sub_ds) in sub_dic.items():
        count = 0
        qi_map = 0
        for i, sub_d in enumerate(sub_ds):
            sub_k, sub_v = list(sub_d.items())[0]
            if sub_v != 0:
                count += 1
                qi_map += float(count) / (i + 1)
                # print(sub_q, sub_d, count, i + 1, float(count) / (i + 1), qi_map)
        q_map.append(qi_map / count)
        # print(qi_map, count, qi_map / count)

    # print(sum(q_map), len(sol_dic), sum(q_map) / len(sol_dic))
    return (sum(q_map) / len(sol_dic))


if __name__ == '__main__':
    debug1 = True
    sub_path = '../data/2_3_1_4_hw01_answer'
    sol_path = '../data/solution_HW2.txt'
    start_index = 1  # !!
    score = mapFun(sub_path, sol_path, start_index)
    print("The MAP score is %.6f" % score)
