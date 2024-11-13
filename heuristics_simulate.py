import random
import copy
import numpy as np
from singal_job_generate import DataTransfer
import time

# 定义变量
num_time_shots = 100  # 假设有5个时间片
# A = sys.float_info.min  # 足够小的常数
A = 0.0000001
B = 100000
b_link = 5  # 每一条链路的带宽
B_port = 40
N = 1000
b_N = 0.04
T_generate = 8
lambda_t = 0
max_server = 64
b_single_port = 40

B_tor_in = [48 for i5 in range(0, 64)]
B_tor_out = [48 for i6 in range(0, 64)]
B_oxc = [96 for i7 in range(0, 64)]
N_oxc = [12 for i8 in range(0, 64)]
d_load_threshold = [1 for i2 in range(0, 100)]
d_add_unload_threshold = [1 for i3 in range(0, 100)]
d_add_load_threshold = [1 for i4 in range(0, 100)]


# def job_generate(rack, data_unload, data_load, node_unload, node_load):
#     """
#     在机架上放置业务，获得非卸载业务的数据传输矩阵和卸载业务的数据输入矩阵
#     :param rack: 机架数目
#     :param data_unload:数组，用来存放所有非卸载业务的worker向ps传输的数据量
#     :param data_load:数组，用来存放所有卸载业务的worker向ps传输的数据量
#     :param node_unload:列表，列表元素为元组，用来存放所有非卸载业务的ps和worker数目
#     :param node_load:列表，用来存放所有卸载业务的worker数目
#     :return:返回非卸载业务的数据传输矩阵和卸载业务的数据输入矩阵
#     """
#     job_DTM = []
#     job_DIM = []
#     Vt_job = []
#     Vi_job = []
#     job_no_INC = []
#     V_j = []
#     for i in range(0, len(data_unload)):
#         unload = DataTransfer(rack, data_unload[i], node_unload[i][0], node_unload[i][1])
#         (A, V_t) = unload.job_unload()
#         job_DTM.append(1 * A)
#         Vt_job.append(1 * V_t)
#         job_no_INC.append(1 * A)
#         V_j.append(1 * V_t)
#     for j in range(0, len(data_load)):
#         load = DataTransfer(rack, data_load[j], 0, node_load[j])
#         (B_1, V_i_1) = load.job_load()
#         ps_local = random.randint(0, 3)
#         B_2 = np.zeros([num_rack, num_rack])
#         V_i_2 = np.zeros([num_rack, num_rack])
#         nonzero_indices = np.nonzero(1 * B_1)
#         first_nonzero_index = nonzero_indices[0][0]
#         B_2[first_nonzero_index][ps_local] += data_load[j] * node_load[j]
#         V_i_2[first_nonzero_index][ps_local] += node_load[j]
#         job_DIM.append(1 * B_1)
#         Vi_job.append(1 * V_i_1)
#         job_no_INC.append(1 * B_2)
#         V_j.append(1 * V_i_2)
#     return job_DTM, job_DIM, Vt_job, Vi_job, job_no_INC, V_j


def sum_port(num_rack, D_T, D_I):
    """
    计算各端口的流量大小
    :param D_T: 非INC业务的数据传输矩阵
    :param D_I: INC业务的数据传输矩阵
    :return: 各端口流量大小
    """
    D_T_sum = sum(D_T)
    D_I_sum = sum(D_I)
    sum_input = [0 for i in range(0, num_rack)]
    sum_output = [0 for i in range(0, num_rack)]
    sum_oxc = [0 for i in range(0, num_rack)]
    if len(D_T) == 0:
        if len(D_I) == 0:
            return sum_input, sum_output, sum_oxc
        else:
            return D_I_sum, np.zeros_like(D_I_sum), np.zeros_like(D_I_sum)
    else:
        sum_input = np.sum(D_T_sum, axis=1) + D_I_sum
        sum_output = np.sum(D_T_sum, axis=0)
        sum_oxc = np.sum(D_T_sum, axis=1) + np.sum(D_T_sum, axis=0) - 2 * np.diag(D_T_sum)
        return sum_input, sum_output, sum_oxc


def sort_port(num_rack, D_T, D_I):
    """
    为端口流量大小排序
    :param D_T: 非INC业务的数据传输矩阵
    :param D_I: INC业务的数据传输矩阵
    :return: 每个端口的流量大小排序
    """
    sum_input, sum_output, sum_oxc = sum_port(num_rack, D_T, D_I)
    return np.argsort(np.argsort(np.concatenate((sum_input, sum_output, sum_oxc))))


def relate_job(num_rack, port, V_t, V_i):
    """
    找出某端口相关所有业务
    :param port: 需要寻找的端口
    :param V_t: 非INC业务的worker分布
    :param V_i: INC业务的worker分布
    :return: 端口相关业务
    """
    r_T = []
    r_I = []
    (kind, num) = (int(port / num_rack), port % num_rack)
    if kind == 0:
        for i in range(0, len(V_t)):
            if np.sum(V_t[i], axis=1)[num] > 0:
                r_T.append(i)
        for j in range(0, len(V_i)):
            if V_i[j][num] > 0:
                r_I.append(j)
    if kind == 1:
        for i in range(0, len(V_t)):
            if np.sum(V_t[i], axis=0)[num] > 0:
                r_T.append(i)
    if kind == 2:
        for i in range(0, len(V_t)):
            if np.sum(V_t[i], axis=1)[num] + np.sum(V_t[i], axis=0)[num] - 2 * np.diag(V_t[i])[num] > 0:
                r_T.append(i)
    return r_T, r_I


def related_not_finish(num_rack, D_T, D_I, V_t, V_i, port):
    """
    除去已经传输完毕的业务
    :param D_T: 非INC业务的数据传输矩阵
    :param D_I: INC业务的数据传输矩阵
    :param V_t: 非INC业务的worker分布
    :param V_i: INC业务的worker分布
    :param port: 需要寻找的端口
    :return: 未传输完的业务
    """
    rT, rI = relate_job(num_rack, port, V_t, V_i)
    rT_x = [] + rT
    rI_x = [] + rI
    for i in rT:
        if np.sum(D_T[i]) == 0:
            rT_x.remove(i)
    for j in rI:
        if np.sum(D_I[j]) == 0:
            rI_x.remove(j)
    return rT_x, rI_x


def max_ideal_add_FCT(PS_time, Delta_t, V_t, V_i, B_T, B_I, r_T, r_I, D_T, D_I, S_T, S_I):
    T_t = np.zeros(len(V_t))
    T_i = np.zeros(len(V_i))
    for i in r_T:
        B_add_T = np.zeros_like(B_T[i]) + B_T[i] + V_t[i]
        if np.sum(np.ceil((sum(B_T) + V_t[i]) * b_link / B_port) - np.ceil(sum(B_T) * b_link / B_port)):
            flag = 1
        else:
            flag = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            FCT_T = np.where(B_add_T != 0, np.true_divide(D_T[i], B_add_T), 0) + flag * Delta_t
        T_t[i] = np.max(FCT_T) + PS_time[S_T[i]]
    for j in r_I:
        B_add_I = np.zeros_like(B_I[j]) + B_I[j] + V_i[j]
        with np.errstate(divide='ignore', invalid='ignore'):
            FCT_I = np.where(B_add_I != 0, np.true_divide(D_I[j], B_add_I), 0)
        T_i[j] = np.max(FCT_I)
    if len(T_t) != 0:
        T_t_max, index_t = np.max(T_t), np.argmax(T_t)
    else:
        T_t_max, index_t = -1, -1
    if len(T_i) != 0:
        T_i_max, index_i = np.max(T_i), np.argmax(T_i)
    else:
        T_i_max, index_i = -1, -1
    if T_t_max != -1 and T_i_max != -1:
        if T_t_max > T_i_max:
            return 0, T_t_max, index_t
        else:
            return 1, T_i_max, index_i
    elif T_i_max == -1:
        return 0, T_t_max, index_t
    else:
        return 1, T_i_max, index_i


def limit_add_min(num_rack, r_T, r_I, B_T, B_I, V_t, V_i):
    rT_x = [] + r_T
    rI_x = [] + r_I
    for i in r_T:
        B_T_add_min = np.zeros_like(B_T) + B_T
        B_I_add_min = np.zeros_like(B_I) + B_I
        B_T_add_min[i] += V_t[i]
        B_T_sum = np.zeros([num_rack, num_rack])
        for j in range(0, len(B_T)):
            B_T_sum += B_T[j]
        B_T_sum += V_t[i]
        P_sum = np.ceil(B_T_sum / b_single_port)
        P_rack = np.zeros(num_rack)
        for j in range(0, num_rack):
            for k in range(0, num_rack):
                if j != k:
                    P_rack[j] += P_sum[j][k] + P_sum[k][j]
        b_port = np.concatenate(sum_port(num_rack, B_T_add_min, B_I_add_min))
        all_less_than_20_in = all(b_port[x] <= B_tor_in[x] for x in range(0, num_rack))
        all_less_than_20_out = all(b_port[x + num_rack] <= B_tor_out[x] for x in range(0, num_rack))
        all_less_than_20_oxc = all(P_rack[x] <= N_oxc[x] for x in range(0, num_rack))
        if all_less_than_20_in is False or all_less_than_20_out is False or all_less_than_20_oxc is False:
            rT_x.remove(i)
    for j in r_I:
        B_T_add_min = np.zeros_like(B_T) + B_T
        B_I_add_min = np.zeros_like(B_I) + B_I
        B_I_add_min[j] += V_i[j]
        b_port = np.concatenate(sum_port(num_rack, B_T_add_min, B_I_add_min))
        all_less_than_20_in = all(b_port[x] <= B_tor_in[x] for x in range(0, num_rack))
        if all_less_than_20_in is False:
            rI_x.remove(j)
    return rT_x, rI_x


def zero_finish(D_T, D_I, B_T, B_I):
    for i in range(0, len(D_T)):
        if np.sum(D_T[i]) == 0:
            B_T[i] = np.zeros_like(D_T[i])
    for j in range(0, len(D_I)):
        if np.sum(D_I[j]) == 0:
            B_I[j] = np.zeros_like(D_I[j])
    return B_T, B_I


def inc_allocate(C, num_rack, W_t, begin, end, num_worker, INC):
    INC_job = []
    num_INC_job = np.zeros(num_rack)
    for i in range(0, len(W_t)):
        if begin[i] == 2 and end[i] == 0:  # 2 表示业务开始并使用 INC
            index = np.argmax(np.sum(W_t[i], axis=1))
            num_INC_job[index] += 1
    for i in range(0, num_rack):
        r = []
        w = []
        inc_rack = []
        for j in range(0, len(W_t)):
            if np.size(W_t[j]) != 1 and np.count_nonzero(W_t[j]) == 1 and begin[j] == 0 and i == np.argmax(
                    np.sum(W_t[j], axis=1)):
                r.append(j)
                w.append(num_worker[j])
        while 1:
            if num_INC_job[i] == C or r == []:
                break
            else:
                if INC == 1:
                    index_w = np.argmax(w)
                else:
                    index_w = r[0]
                inc_rack.append(r[index_w])
                del r[index_w]
                del w[index_w]
                num_INC_job[i] += 1
        INC_job += inc_rack
    return INC_job


def Transport_matrix(num_rack, D_T, D_I, V_t, V_i, B_T, B_I, S_T, S_I, W_t, begin, INC_Ts, data):
    Delete_T = []
    for i in range(0, len(D_T)):
        if begin[S_T[i]] == 0:
            Delete_T.append(i)
    D_T = [x for i, x in enumerate(D_T) if i not in Delete_T]
    V_t = [x for i, x in enumerate(V_t) if i not in Delete_T]
    S_T = [x for i, x in enumerate(S_T) if i not in Delete_T]
    B_T = [x for i, x in enumerate(B_T) if i not in Delete_T]
    Delete_I = []
    for i in range(0, len(D_I)):
        if begin[S_I[i]] == 0:
            Delete_I.append(i)
    D_I = [x for i, x in enumerate(D_I) if i not in Delete_I]
    V_i = [x for i, x in enumerate(V_i) if i not in Delete_I]
    S_I = [x for i, x in enumerate(S_I) if i not in Delete_I]
    B_I = [x for i, x in enumerate(B_I) if i not in Delete_I]
    for i in range(0, len(W_t)):
        if begin[i] != 0 or np.size(W_t[i]) == 1:
            continue
        if i in INC_Ts:
            D_I.append(np.sum(W_t[i], axis=1) * data[i])
            V_i.append(np.sum(W_t[i], axis=1))
            S_I.append(i)
            B_I.append(np.zeros(num_rack))
        else:
            D_T.append(W_t[i] * data[i])
            V_t.append(W_t[i])
            S_T.append(i)
            B_T.append(np.zeros([num_rack, num_rack]))
    return D_T, D_I, V_t, V_i, B_T, B_I, S_T, S_I


def allocate(PS_time, C, num_rack, Delta_t, Ts, data, D_T, D_I, V_t, V_i, B_T, B_I, S_T, S_I, begin_no_INC, begin_INC, num_worker, new_job_Ts, begin_job, end, end_allocate, INC, algo, W_t):
    B_T, B_I = zero_finish(D_T, D_I, B_T, B_I)
    B_all = 0
    num_port = 3 * num_rack
    for i in range(0, len(begin_job)):
        if begin_job[i] == 1:
            if np.sum(D_T[S_T.index(i)]) == 0:
                end[i] = 1
        if begin_job[i] == 2:
            if np.sum(D_I[S_I.index(i)]) == 0:
                end[i] = 1
    begin_no_INC = []
    begin_INC = []
    end_allocate_no_INC = []
    end_allocate_INC = []
    end_allocate += [0 for i in range(len(begin_job) - len(end_allocate))]
    for i in range(0, len(begin_job)):
        if begin_job[i] == 1:
            begin_no_INC.append(1)
            if end_allocate[i] == 1:
                end_allocate_no_INC.append(1)
        if begin_job[i] == 2:
            begin_INC.append(1)
            if end_allocate[i] == 1:
                end_allocate_INC.append(1)
    W_t = worker_local(num_rack, num_worker, begin_job, end, new_job_Ts, W_t)
    if INC == 1:
        INC_Ts = inc_allocate(C, num_rack, W_t, begin_job, end, num_worker, INC)
    else:
        INC_Ts = []
    D_T, D_I, V_t, V_i, B_T, B_I, S_T, S_I = Transport_matrix(num_rack, D_T, D_I, V_t, V_i, B_T, B_I, S_T, S_I, W_t, begin_job, INC_Ts, data)
    if algo == 0:
        for i in range(0, len(begin_job)):
            flag = 0
            if begin_job[i] == 0:
                if np.size(W_t[i]) == 1:
                    break
                rT = []
                rI = []
                if i in S_T:
                    rT.append(S_T.index(i))
                if i in S_I:
                    rI.append(S_I.index(i))
                rT, rI = limit_add_min(num_rack, rT, rI, B_T, B_I, V_t, V_i)
                if rT == [] and rI == []:
                    break
                while 1:
                    rT, rI = limit_add_min(num_rack, rT, rI, B_T, B_I, V_t, V_i)
                    if rT == [] and rI == []:
                        break
                    if rT:  # 非 INC 业务
                        index = S_T.index(i)
                        B_T[index] += V_t[index]
                        begin_job[S_T[index]] = 1
                        B_all += np.sum(V_t[index])
                        continue
                    elif rI:  # INC 业务
                        index = S_I.index(i)
                        B_I[index] += V_i[index]
                        begin_job[S_I[index]] = 2
                        B_all += np.sum(V_i[index])
                        continue
    if algo == 1:
        s = np.argsort(np.argsort(ideal_PCT_new(PS_time, num_rack, D_T, D_I, V_t, V_i, S_T, S_I)))
        for p in range(0, num_port):
            port = np.where(s == num_port - p - 1)[0][0]  # 按序遍历 port
            rT, rI = related_not_finish(num_rack, D_T, D_I, V_t, V_i, port)  # 找到相关业务
            while 1:
                rT, rI = limit_add_min(num_rack, rT, rI, B_T, B_I, V_t, V_i)  # 筛出带宽溢出的业务
                if Ts != 0:
                    for i in range(0, len(end_allocate_no_INC)):
                        if end_allocate_no_INC[i] == 1:
                            if i in rT:
                                rT.remove(i)
                    for i in range(0, len(end_allocate_INC)):
                        if end_allocate_INC[i] == 1:
                            if i in rI:
                                B_I[i] = V_i[i]
                if rT == [] and rI == []:
                    break
                flag, t, index = max_ideal_add_FCT(PS_time, Delta_t, V_t, V_i, B_T, B_I, rT, rI, D_T, D_I, S_T, S_I)
                if flag == 0:  # 非 INC 业务
                    B_T[index] += V_t[index]
                    begin_job[S_T[index]] = 1
                    continue
                elif flag == 1:  # INC 业务
                    B_I[index] += V_i[index]
                    begin_job[S_I[index]] = 2
                    continue
    index_t = []
    index_i = []
    for i in range(0, len(begin_job)):
        if begin_job[i] == 0:
            if i in S_T:
                index_t.append(S_T.index(i))
            if i in S_I:
                index_i.append(S_I.index(i))
    D_T = [x for i, x in enumerate(D_T) if i not in index_t]
    V_t = [x for i, x in enumerate(V_t) if i not in index_t]
    S_T = [x for i, x in enumerate(S_T) if i not in index_t]
    B_T = [x for i, x in enumerate(B_T) if i not in index_t]
    D_I = [x for i, x in enumerate(D_I) if i not in index_i]
    V_i = [x for i, x in enumerate(V_i) if i not in index_i]
    S_I = [x for i, x in enumerate(S_I) if i not in index_i]
    B_I = [x for i, x in enumerate(B_I) if i not in index_i]
    for i in range(0, len(begin_job)):
        if begin_job[i] > 0:
            end_allocate[i] = 1
    return D_T, D_I, B_T, B_I, V_t, V_i, S_T, S_I


def increase_T(B_T, B_T_pre):
    B_delta = []
    for i in range(0, len(B_T)):
        B_delta.append(B_T[i] - B_T_pre[i])
    for i in range(0, len(B_T)):
        B_delta[i][B_delta[i] < 0] = 0
        np.fill_diagonal(B_delta[i], 0)
    return B_delta


def port_count(num_rack, B_T):
    B_sum = np.zeros([num_rack, num_rack])
    for i in range(0, len(B_T)):
        B_sum += B_T[i]
    P = np.ceil(B_sum / b_single_port)
    P_used = np.zeros([num_rack, num_rack])
    for i in range(0, num_rack):
        P[i][i] = 0
        for j in range(0, num_rack):
            if i != j:
                P_used[i][j] += P[i][j] + P[j][i]
            else:
                P_used[i][j] = 0
    return P, P_used


def transport(PS_time, C, num_rack, Delta_t, Ts, data, D_T, D_I, B_T, B_I, V_t, V_i, S_T, S_I, begin_no_INC, begin_INC, num_worker, new_job_Ts, non_begin, end, end_allocate, INC, algo, W_t):
    B_T_pre = copy.deepcopy(B_T)
    D_T, D_I, B_T, B_I, V_t, V_i, S_T, S_I = allocate(PS_time, C, num_rack, Delta_t, Ts, data, D_T, D_I, V_t, V_i, B_T, B_I, S_T, S_I, begin_no_INC, begin_INC, num_worker, new_job_Ts, non_begin, end, end_allocate, INC, algo, W_t)
    if len(B_T_pre) < len(B_T):
        for i in range(0, len(B_T) - len(B_T_pre)):
            B_T_pre.append(np.zeros([num_rack, num_rack]))

    B_T_increase = increase_T(B_T, B_T_pre)
    Reconfigure = np.zeros(len(B_T))
    P1, P1_used = port_count(num_rack, B_T_pre)
    P2, P2_used = port_count(num_rack, B_T)
    Delta_P = P2 - P1
    Reconfigure_t = 0
    for i in range(0, num_rack):
        for j in range(0, num_rack):
            if Delta_P[i][j] >= 1:
                if P2_used[i][j] - P1_used[i][j] > 0:
                    Reconfigure_t = 1
                for k in range(0, len(B_T)):
                    if B_T_increase[k][i][j] > 0 and P2_used[i][j] - P1_used[i][j] > 0 and P2[i][j] - P1[i][j] > 0:
                        Reconfigure[k] = 1
    for i in range(0, len(D_T)):
        if np.sum(D_T) == 0:
            end[S_T[i]] = 1
        for a in range(0, num_rack):
            for b in range(0, num_rack):
                if a == b:
                    D_T[i][a][b] = max(0, D_T[i][a][b] - B_T[i][a][b] * Delta_t * b_link)
                else:
                    D_T[i][a][b] = max(0, D_T[i][a][b] - B_T[i][a][b] * Delta_t * b_link + B_T_increase[i][a][
                        b] * Delta_t * b_link * Reconfigure[i])
    for j in range(0, len(D_I)):
        if np.sum(D_I) == 0:
            end[S_I[j]] = 1
        for c in range(0, num_rack):
            D_I[j][c] = max(0, D_I[j][c] - B_I[j][c] * Delta_t * b_link)
    return D_T, D_I, B_T, B_I, V_t, V_i, S_T, S_I, Reconfigure_t


def ideal_PCT_new(PS_time, num_rack, D_T, D_I, V_t, V_i, S_T, S_I):
    num_port = 3 * num_rack
    PCT = np.zeros(num_port)
    if D_T == [] and D_I == []:
        return PCT
    data_input, data_output, data_oxc = sum_port(num_rack, D_T, D_I)
    data = np.concatenate((data_input, data_output, data_oxc))
    for p in range(0, num_port):
        if p < num_rack:
            t_transport = data[p] / (B_tor_in[p] * b_link)
        elif p < 2 * num_rack:
            t_transport = data[p] / (B_tor_out[p - num_rack] * b_link)
        else:
            t_transport = data[p] / (N_oxc[p - 2 * num_rack] * B_port)
        rT, rI = related_not_finish(num_rack, D_T, D_I, V_t, V_i, p)
        t_calculate = 0
        for i in rT:
            if PS_time[S_T[i]] > t_calculate:
                t_calculate = PS_time[S_T[i]]
        T = t_transport + t_calculate
        min_T = t_transport
        max_T = T
        while max_T - min_T >= 0.01:
            mid = (max_T + min_T) / 2
            B_rT = np.zeros(len(rT))
            B_rI = np.zeros(len(rI))
            count_T = 0
            count_I = 0
            flag = 0
            for i in rT:
                if mid - PS_time[S_T[i]] == 0:
                    flag = 1
                    break
                if p < num_rack:
                    B_rT[count_T] = np.sum(D_T[i], axis=1)[p] / (mid - PS_time[S_T[i]])
                elif p < 2 * num_rack:
                    B_rT[count_T] = np.sum(D_T[i], axis=0)[p - num_rack] / (mid - PS_time[S_T[i]])
                else:
                    B_rT[count_T] = (np.sum(D_T[i], axis=1)[p - 2 * num_rack] + np.sum(D_T[i], axis=1)[
                        p - 2 * num_rack] - 2 * D_T[i][p - 2 * num_rack][p - 2 * num_rack]) / (
                                            mid - PS_time[S_T[i]])
                count_T += 1
            for j in rI:
                B_rI[count_I] += D_I[j][p] / mid
                count_I += 1
            B_all = np.sum(B_rT) + np.sum(B_rI)
            if flag == 1:
                min_T = mid
            if p < num_rack:
                if B_all <= B_tor_in[p] * b_link:
                    max_T = mid
                else:
                    min_T = mid
            elif p < 2 * num_rack:
                if B_all <= B_tor_out[p - num_rack] * b_link:
                    max_T = mid
                else:
                    min_T = mid
            else:
                if B_all <= B_oxc[p - 2 * num_rack] * b_link:
                    max_T = mid
                else:
                    min_T = mid
        PCT[p] = max_T
    return PCT


def worker_local(num_rack, num_worker, begin, end, new_worker, W_t):
    use_server = np.zeros(num_rack)
    pending_job = []
    pending_job_worker = []
    for i in range(0, len(W_t)):
        if begin[i] == 1 and end[i] == 0 and np.size(W_t[i]) != 1:
            use_server += np.sum(W_t[i], axis=1)
            index = np.argmax(np.sum(W_t[i], axis=0))
            use_server[index] += 1
        if begin[i] == 2 and end[i] == 0 and np.size(W_t[i]) != 1:
            use_server += np.sum(W_t[i], axis=1)
        if begin[i] == 0 and np.size(W_t[i]) == 1:
            pending_job.append(i)
            pending_job_worker.append(num_worker[i])
    for i in range(0, len(new_worker)):
        pending_job.append(i + len(W_t))
        pending_job_worker.append(num_worker[i + len(W_t)])
    sort_pending_job = []
    length = len(pending_job)
    for i in range(0, length):
        index = pending_job_worker.index(max(pending_job_worker))
        sort_pending_job.append(pending_job[index])
        del pending_job[index]
        del pending_job_worker[index]
    pend = len(sort_pending_job)
    W_t += [0 for i in range(0, len(new_worker))]
    for p in range(0, pend):
        i = sort_pending_job[p]
        if np.size(W_t[i]) != 1:
            continue
        if np.sum(use_server) + num_worker[i] + 1 > num_rack * max_server:
            continue
        local_worker = np.zeros(num_rack)
        remind_worker = num_worker[i]
        count = 0
        test_server = copy.deepcopy(use_server)
        local_ps = np.argmin(test_server)
        test_server[local_ps] += 1
        while 1:
            index_r = np.argmin(test_server)
            if index_r != local_ps:
                count += 1
            if count > N_oxc[local_ps]:
                flag = 0
                break
            if remind_worker <= max_server - test_server[index_r]:
                local_worker[index_r] += remind_worker
                remind_worker = 0
                flag = 1
                break
            if remind_worker > max_server - test_server[index_r]:
                local_worker[index_r] += max_server - test_server[index_r]
                remind_worker -= max_server - test_server[index_r]
                test_server[index_r] = max_server
        if flag == 0:
            W_t[i] = 0
        elif flag == 1:
            W = np.zeros([num_rack, num_rack])
            for j in range(0, num_rack):
                W[j][local_ps] += local_worker[j]
                W_t[i] = W
            use_server += np.sum(W, axis=1)
            index = np.argmax(np.sum(W, axis=0))
            use_server[index] += 1
    return W_t


def schedule(PS_time, C, num_rack, Delta_t, data, init_job, new_job, num_worker, S_T, S_I, end_allocate, INC, algo):
    count = 0
    oxc_delta = 0
    Bandwidth = 0
    count_reconfigure = 0
    W_t = []
    D_T = []
    D_I = []
    V_t = []
    V_i = []
    B_T = [np.zeros([num_rack, num_rack]) for i in range(0, len(D_T))]
    B_I = [np.zeros(num_rack) for i in range(0, len(D_I))]
    T_t = np.zeros(len(D_T))
    T_i = np.zeros(len(D_I))
    T = []
    non_begin_no_INC = []  # 1表示已经开始，0表示未开始
    non_begin_INC = []
    non_begin = []  # 2表示已经开始且为INC，1表示已经开始且为非INC，0表示未开始
    end = []
    for i in range(0, len(V_t) + len(V_i)):
        S = np.concatenate((S_T, S_I))
        index = np.where(S == i)[0][0]
        if index < len(S_T):
            non_begin.append(1)
        else:
            non_begin.append(2)
    k = 0
    while 1:
        if k < len(new_job):
            T = np.concatenate((T, [k * Delta_t for i in range(0, int(new_job[k]))]))
            non_begin = np.concatenate((non_begin, [0 for i in range(0, int(new_job[k]))]))
            end = np.concatenate((end, [0 for i in range(0, int(new_job[k]))]))
            new = [num_worker[int(count) + i] for i in range(0, int(new_job[k]))]
            count += new_job[k]
        else:
            new = []
        for i in range(0, len(T)):
            if non_begin[i] == 0:
                T[i] += Delta_t
            elif non_begin[i] == 1:
                x = np.count_nonzero(non_begin[: i] == 1)
                if np.sum(D_T[x]) != 0:
                    T[i] += Delta_t
            else:
                x = np.count_nonzero(non_begin[: i] == 2)
                if np.sum(D_I[x]) != 0:
                    T[i] += Delta_t
        if np.sum(D_T) == 0 and np.sum(D_I) == 0:
            if len([x for x in non_begin if x != 0]) == len(num_worker):
                break
        D_T, D_I, B_T, B_I, V_t, V_i, S_T, S_I, Delta = transport(PS_time, C, num_rack, Delta_t, k, data, D_T, D_I, B_T, B_I, V_t, V_i, S_T, S_I,
                                                     non_begin_no_INC, non_begin_INC, num_worker, new, non_begin, end,
                                                     end_allocate, INC, algo, W_t)
        oxc_delta += Delta
        k += 1
    for i in range(0, len(T)):
        if non_begin[i] == 1:
            index = S_T.index(i)
            T[i] += PS_time[i]
    t = max(T)
    print("oxc重构次数：", oxc_delta, "; t = ", t)
    # routine = "output" + str(Delta_t) + ".txt"
    # with open(routine, 'a') as file1:
    #     file1.write("oxc重构量：" + str(oxc_delta) + " t = " + str(t) + '\n')
    return t
