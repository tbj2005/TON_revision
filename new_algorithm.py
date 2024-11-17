import numpy as np
import copy
from scipy.optimize import linear_sum_assignment


def job_exit(rack_num, local_solution, reserve_server, add_end, inc_usage, inc_reserve):
    for i in add_end:
        worker = np.sum(local_solution[i], axis=1)
        ps = np.sum(local_solution[i], axis=0)
        for r in range(0, rack_num):
            reserve_server[r] -= worker[r]
            if ps[r] > 0:
                reserve_server[r] -= 1
            if inc_usage[r][i] == 1:
                inc_reserve[r] -= 1
    return reserve_server, inc_reserve


def job_deploy(local_solution, reserve_server, new_job, job_worker_num, data_matrix, data_per_worker):
    for i in new_job:
        if job_worker_num[i] + 1 <= np.sum(reserve_server):
            ps_local = np.argmax(reserve_server)
            reserve_server[ps_local] -= 1
            worker_count = job_worker_num[i]
            while worker_count > 0:
                index = np.argmax(reserve_server)
                worker_index = min(reserve_server[index], worker_count)
                worker_count -= worker_index
                local_solution[i][index][ps_local] += worker_index
                reserve_server[index] -= worker_index
                data_matrix[i][index][ps_local] += worker_index * data_per_worker[i]
        else:
            return -1, -1, -1
    return local_solution, reserve_server, data_matrix


def benefit_cal(new_job, local_solution, inc_benefit, rack_num, d_per_worker):
    for i in new_job:
        ps = np.sum(local_solution[i], 0)
        ps_local = np.argmax(ps)
        for j in range(0, rack_num):
            inc_benefit[i][j] = local_solution[i][j][ps_local] * d_per_worker[i]
    return inc_benefit


def inc_allocate(inc_reserve, inc_benefit, inc_job, job_wait, rack_num, inc_usage):
    inc_benefit_copy = inc_benefit[0:job_wait]
    while all(inc_benefit_copy <= 0) == 0 and np.sum(inc_reserve) > 0:
        row = [i for i in range(0, rack_num) if inc_reserve[i] == 0]
        col = [i for i in range(0, len(inc_job)) if i not in job_wait]
        for i in row:
            inc_benefit_copy[i, :] = -1
        for i in col:
            inc_benefit_copy[:, i] = -1
        row_ch, col_ch = linear_sum_assignment(inc_benefit_copy, True)
        for i in row_ch:
            for j in col_ch:
                if inc_benefit_copy[i][j] > 0:
                    inc_usage[i][j] = 1
                    inc_reserve[i] -= 1
                    inc_benefit_copy[:, j] = -1
                    inc_benefit[:, j] = -1
                    inc_job[i] = 1
    return inc_usage, inc_reserve, inc_benefit, inc_job


def data_obtain(inc_usage, rack_num, d_per_worker, d_matrix, new_job, local_solution):
    for i in new_job:
        ps = np.sum(local_solution[i], 0)
        ps_local = np.argmax(ps)
        for j in range(0, rack_num):
            if inc_usage[i][j] == 0:
                d_matrix[i][j][ps_local] = local_solution[i][j][ps_local] * d_per_worker[i]
            else:
                if ps_local == j:
                    d_matrix[i][j][ps_local] = local_solution[i][j][ps_local] * d_per_worker[i]
                else:
                    d_matrix[i][j][ps_local] = d_per_worker[i]
    return d_matrix


def pct_rack(inc_usage, d_matrix, rack_index, local_solution, rack_num, job_agg, b_tor, b_per_port, port_per_rack):
    relate_job_up = []
    traffic_up = []
    agg_up = []
    relate_job_down = []
    traffic_down = []
    agg_down = []
    relate_job_oxc_up = []
    traffic_oxc_up = []
    agg_oxc_up = []
    relate_job_oxc_down = []
    traffic_oxc_down = []
    agg_oxc_down = []
    for i in range(0, len(d_matrix)):
        if sum([d_matrix[i][rack_index][j] for j in range(0, rack_num)]) > 0:
            relate_job_up.append(i)
            if inc_usage[i][rack_index] == 1:
                traffic_up.append(sum([d_matrix[i][rack_index][j] for j in range(0, rack_num)]) * local_solution[i][rack_index])
            else:
                traffic_up.append(sum([d_matrix[i][rack_index][j] for j in range(0, rack_num)]))
            if np.count_nonzero(local_solution[i]) > 1:
                agg_up.append(job_agg[i])
            elif sum([inc_usage[i][j] for j in range(0, rack_num)]) == 0:
                agg_up.append(job_agg[i])
            else:
                agg_up.append(0)
        if sum([d_matrix[i][j][rack_index] for j in range(0, rack_num)]) > 0:
            relate_job_down.append(i)
            if inc_usage[i][rack_index] == 1:
                traffic_down.append(0)
            else:
                traffic_down.append(sum([d_matrix[i][j][rack_index] for j in range(0, rack_num)]))
            if np.count_nonzero(local_solution[i]) > 1:
                agg_down.append(job_agg[i])
            elif sum([inc_usage[i][j] for j in range(0, rack_num)]) == 0:
                agg_down.append(job_agg[i])
            else:
                agg_down.append(0)
        if sum([d_matrix[i][rack_index][j] for j in range(0, rack_num) if j != rack_index]) > 0:
            relate_job_oxc_up.append(i)
            traffic_oxc_up.append(sum([d_matrix[i][rack_index][j] for j in range(0, rack_num) if j != rack_index]))
            if np.count_nonzero(local_solution[i]) > 1:
                agg_oxc_up.append(job_agg[i])
            elif sum([inc_usage[i][j] for j in range(0, rack_num)]) == 0:
                agg_oxc_up.append(job_agg[i])
            else:
                agg_oxc_down.append(0)
        if sum([d_matrix[i][j][rack_index] for j in range(0, rack_num) if j != rack_index]) > 0:
            relate_job_oxc_down.append(i)
            traffic_oxc_down.append(sum([d_matrix[i][j][rack_index] for j in range(0, rack_num) if j != rack_index]))
            if np.count_nonzero(local_solution[i]) > 1:
                agg_oxc_down.append(job_agg[i])
            elif sum([inc_usage[i][j] for j in range(0, rack_num)]) == 0:
                agg_oxc_down.append(job_agg[i])
            else:
                agg_oxc_down.append(0)


    # 上行：
    threshold = 0.01
    pct_up = bisection(relate_job_up, agg_up, traffic_up, b_tor[rack_index], threshold)
    pct_down = bisection(relate_job_down, agg_down, traffic_down, b_tor[rack_index], threshold)
    pct_oxc_up = bisection(relate_job_oxc_up, agg_oxc_up, traffic_oxc_up, b_per_port * port_per_rack, threshold)
    pct_oxc_down = bisection(relate_job_oxc_down, agg_oxc_down, traffic_oxc_down, b_per_port * port_per_rack, threshold)
    return (pct_up, pct_down, pct_oxc_up, pct_oxc_down, relate_job_up, traffic_up, agg_up, relate_job_down, traffic_down,
            agg_down, relate_job_oxc_up, traffic_oxc_up, agg_oxc_up, relate_job_oxc_down, traffic_oxc_down, agg_oxc_down)



def bisection(relate, agg, traffic, bandwidth, threshold):
    t_max_up = max(agg) + sum(traffic) / bandwidth
    t_min_up = sum(traffic) / bandwidth
    while t_max_up - t_min_up >= threshold:
        t_mid_up = (t_max_up + t_min_up) / 2
        if sum([traffic[i] / (t_mid_up - agg[i]) for i in relate]) > bandwidth:
            t_max_up = t_mid_up
        else:
            t_min_up = t_mid_up
    return t_min_up


def pct(inc_usage, d_matrix, rack_num, local_solution, job_agg, b_tor, b_oxc_port, port_per_rack):
    port_t = np.array([0 for i in range(0, 4 * rack_num)])
    relate_up = []
    relate_down = []
    relate_oxc_up = []
    relate_oxc_down = []
    tra_up = []
    tra_down = []
    tra_oxc_up = []
    tra_oxc_down = []
    a_up = []
    a_down = []
    a_oxc_up = []
    a_oxc_down = []
    for i in range(0, rack_num):
        (pct_up, pct_down, pct_oxc_up, pct_oxc_down, relate_job_up, traffic_up, agg_up, relate_job_down, traffic_down,
         agg_down, relate_job_oxc_up, traffic_oxc_up, agg_oxc_up, relate_job_oxc_down, traffic_oxc_down,
         agg_oxc_down) = pct_rack(inc_usage,d_matrix, i, local_solution, rack_num, job_agg, b_tor, b_oxc_port,
                                  port_per_rack)
        port_t[i] += pct_up
        port_t[i + rack_num] += pct_down
        port_t[i + 2 * rack_num] += pct_oxc_up
        port_t[i + 3 * rack_num] += pct_oxc_down
        relate_up.append(relate_job_up)
        relate_down.append(relate_job_down)
        relate_oxc_up.append(relate_job_oxc_up)
        relate_oxc_down.append(relate_job_oxc_down)
        tra_up.append(traffic_up)
        tra_down.append(traffic_down)
        tra_oxc_up.append(traffic_oxc_up)
        tra_oxc_down.append(traffic_oxc_down)
        a_up.append(agg_up)
        a_down.append(agg_down)
        a_oxc_up.append(agg_oxc_up)
        a_oxc_down.append(agg_oxc_down)
    return (port_t, relate_up, tra_up, a_up, relate_down, tra_down, a_down, relate_oxc_up, tra_oxc_up, a_oxc_up,
            relate_oxc_down, tra_oxc_down, a_oxc_down)


def feasible_add(b_per_worker, relate, local_solution, b_tor_in, b_tor_out, b_inter, rack_num, inc_usage, b_tor,
                 b_oxc_port, port_per_rack, agg_time, b_unit, d_matrix):
    t_add = []
    for job_index in relate:
        b_tor_in_test = copy.deepcopy(b_tor_in)
        b_tor_out_test = copy.deepcopy(b_tor_out)
        b_inter_test = copy.deepcopy(b_inter)
        p_inter = np.zeros([rack_num, rack_num])
        for i in range(0, rack_num):
            b_tor_in_test[i] += sum([local_solution[job_index][i][k] for k in range(0, rack_num)]) * b_per_worker[i]
            b_tor_out_test[i] += ((1 - inc_usage[i][job_index]) * b_per_worker[job_index] *
                                  (sum([(inc_usage[k][job_index] + (1 - inc_usage[k][job_index]) *
                                   local_solution[job_index][k][i]) if k != i else local_solution[job_index][i][i] for
                                   k in range(0, rack_num)])))
            for j in range(0, rack_num):
                if i != j:
                    b_inter_test[i][j] += (inc_usage[i][job_index] + (1 - inc_usage[i][job_index]) *
                                           local_solution[job_index][i][j]) * b_per_worker
                    p_inter[i][j] = np.ceil(b_inter_test[i][j] / b_oxc_port)

        in_resource = np.array([b_tor_in_test[i] - b_tor[i] for i in range(0, rack_num)])
        out_resource = np.array([b_tor_out_test[i] - b_tor[i] for i in range(0, rack_num)])
        p_inter_up = [sum([p_inter[i][j] for j in range(0, rack_num)]) for i in range(0, rack_num)]
        p_inter_down = [sum([p_inter[j][i] for j in range(0, rack_num)]) for i in range(0, rack_num)]
        if all(in_resource >= 0) == 0 or all(out_resource >= 0) == 0:
            t_add.append(-1)
        elif max(p_inter_up + p_inter_down) > port_per_rack:
            t_add.append(-1)
        else:
            t_matrix = np.zeros([rack_num, rack_num])
            for i in range(0, rack_num):
                for j in range(0, rack_num):
                    if i != j and d_matrix[i][j] > 0:
                        t_matrix[i][j] = (d_matrix[i][j] / (inc_usage[i][job_index] + (1 - inc_usage[i][job_index]) *
                                          local_solution[job_index][i][j]) * (b_per_worker[job_index] + b_unit) +
                                          agg_time[job_index])
            t_add.append(np.max(t_matrix))
    return t_add


def bandwidth_allocate(d_matrix, job_agg, local_solution, algo, inc_usage, rack_num, b_tor, b_oxc_port, port_per_rack, b_unit, b_per_worker, begin):
    for i in range(0, len(d_matrix)):
        if np.sum(d_matrix[i]) == 0:
            b_per_worker[i] = 0
        if np.sum(d_matrix[i]) > 0 and begin[i] == 1:
            b_per_worker[i] = b_unit
    if algo == 1:
        (port_t, relate_up, tra_up, a_up, relate_down, tra_down, a_down, relate_oxc_up, tra_oxc_up, a_oxc_up,
         relate_oxc_down, tra_oxc_down, a_oxc_down) = pct(inc_usage, d_matrix, rack_num, local_solution, job_agg, b_tor,
                                                          b_oxc_port, port_per_rack)
        b_tor_up = []
        b_tor_down = []
        b_inter = np.zeros([rack_num, rack_num])
        for i in range(0, rack_num):
            b_tor_up.append(sum([sum([local_solution[j][i][k] for k in range(0, rack_num)]) * b_per_worker[j]] for j in
                                range(0, len(d_matrix))))
            b_tor_down.append(sum([(1 - inc_usage[i][j]) * b_per_worker[j] * (sum([(inc_usage[k][j] + (1 -
                              inc_usage[k][j]) * local_solution[j][k][i]) if k != i else local_solution[j][i][i] for k
                              in range(0, rack_num)])) for j in range(0, len(d_matrix))]))
        for i in range(0, rack_num):
            for j in range(0, rack_num):
                if i != j:
                    b_inter[i][j] += sum([(inc_usage[i][k] + (1 - inc_usage[i][k]) * local_solution[k][i][j]) *
                                          b_per_worker[j] for k in range(0, len(d_matrix))])
        sort_port = np.argsort(port_t)
        for i in sort_port:
            if i < rack_num:
                relate_job = relate_up[i]
                traffic = tra_up[i]
                agg_time = a_up[i]
            elif i < 2 * rack_num:
                relate_job = relate_down[i]
                traffic = tra_down[i]
                agg_time = a_down[i]
            elif i < 3 * rack_num:
                relate_job = relate_oxc_up[i]
                traffic = tra_oxc_up[i]
                agg_time = a_oxc_up[i]
            else:
                relate_job = relate_oxc_down[i]
                traffic = tra_oxc_down[i]
                agg_time = a_oxc_down[i]
            while 1:
                t_add = feasible_add(b_per_worker, relate_job, local_solution, b_tor_up, b_tor_down, b_inter,
                                     rack_num, inc_usage, b_tor, b_oxc_port, port_per_rack, agg_time, b_unit,
                                     d_matrix)
                max_index = np.argmax(t_add)
                if t_add[max_index] != -1:
                    b_per_worker[relate_job[max_index]] += b_unit
                    begin[relate_job[max_index]] = 1
                else:
                    break

        b_inter_new = np.zeros([rack_num, rack_num])
        for i in range(0, rack_num):
            for j in range(0, rack_num):
                if i != j:
                    b_inter_new[i][j] += sum([(inc_usage[i][k] + (1 - inc_usage[i][k]) * local_solution[k][i][j]) *
                                             b_per_worker[j] for k in range(0, len(d_matrix))])
        return b_per_worker, begin, b_inter_new
    if algo == 0:
        for i in rack_num(0, len(d_matrix)):
            if


def recon(oxc_topo, b_inter, rack_num, b_oxc_port, b_per_worker_old, b_per_worker, inc_usage, local_solution, begin, end):
    new_oxc_topo = np.zeros([rack_num, rack_num])
    recon_bandwidth = np.zeros(len(b_per_worker))
    for i in range(0, rack_num):
        for j in range(0, rack_num):
            if i > j:
                new_oxc_topo[i][j] = max(b_inter[i][j], b_inter[j][i]) / b_oxc_port
                new_oxc_topo[j][i] = max(b_inter[i][j], b_inter[j][i]) / b_oxc_port
    add_oxc_topo = new_oxc_topo - oxc_topo
    add_oxc_topo = add_oxc_topo[add_oxc_topo > 0]
    for i in range(0, rack_num):
        for j in range(0, rack_num):
            for k in range(0, len(b_per_worker)):
                if add_oxc_topo[i][j] > 0 and begin[k] == 1 and end[k] == 0 and local_solution[k][i][j] > 0 and inc_usage[j][k] == 0:
                    recon_bandwidth[k] = abs(b_per_worker[k] - b_per_worker_old[k])
    return recon_bandwidth


def communication(d_matrix, inc_usage, b_per_worker, recon_bandwidth, local_solution, rack_num, t_recon):
    for i in range(0, len(d_matrix)):
        if b_per_worker[i] > 0:
            for u in range(0, rack_num):
                for v in range(0, rack_num):
                    data_tran = min(b_per_worker[i] * (local_solution[i][u][v] * (1 - inc_usage[u][i]) +
                                    inc_usage[u][i]) - recon_bandwidth[i] * t_recon, d_matrix[i][u][v])
                    d_matrix[i][u][v] -= data_tran
    return d_matrix


def schedule(rack_num, server_per_rack, init_num, job_arrive_time, job_worker_num, job_agg, d_per_worker, algo, ts_len,
             inc_limit, b_tor, b_oxc_port, port_per_rack, b_unit, t_recon):
    job_list = []
    ts_count = 0
    local_solution = []
    reserve_server = np.array([server_per_rack for i in range(0, rack_num)])
    inc_usage = np.zeros([rack_num, len(job_worker_num)])
    inc_reserve = np.array([inc_limit for i in range(0, rack_num)])
    inc_benefit = np.ones([rack_num, len(job_worker_num)])
    inc_job = []
    end = []
    begin = []
    d_matrix = []
    b_per_worker = []
    job_wait = []
    oxc_topo = np.zeros([rack_num, rack_num])
    while 1:
        if ts_count == 0:
            end += [0 for i in range(0, init_num)]
            begin += [0 for i in range(0, init_num)]
            job_list += [i for i in range(0, init_num)]
            inc_job += [0 for i in range(0, init_num)]
            job_wait += [i for i in range(0, init_num)]
            new_job = copy.deepcopy(job_list)
            local_solution += [np.zeros([rack_num, rack_num]) for i in range(0, init_num)]
            d_matrix += [np.zeros([rack_num, rack_num]) for i in range(0, init_num)]
            b_per_worker += [0 for i in range(0, init_num)]
        else:
            new_job = []
            count_job = len(job_list)
            if count_job == len(job_arrive_time):
                break
            if init_num != 0:
                while job_arrive_time[count_job] <= ts_count * ts_len + job_arrive_time[init_num - 1]:
                    new_job.append(count_job)
                    count_job += 1
                    job_list.append(count_job)
                    job_wait.append(count_job)
                    inc_job.append(0)
                    local_solution.append(np.zeros([rack_num, rack_num]))
                    d_matrix.append(np.zeros([rack_num, rack_num]))
                    b_per_worker.append(0)
                    end.append(0)
                    begin.append(0)
                    if count_job == len(job_arrive_time):
                        break
            else:
                while job_arrive_time[count_job] <= ts_count * ts_len:
                    new_job.append(count_job)
                    count_job += 1
                    job_list.append(count_job)
                    job_wait.append(count_job)
                    local_solution.append(np.zeros([rack_num, rack_num]))
                    d_matrix.append(np.zeros([rack_num, rack_num]))
                    b_per_worker.append(0)
                    end.append(0)
                    begin.append(0)
                    inc_job.append(0)
                    if count_job == len(job_arrive_time):
                        break
        print(ts_count, len(job_list), len(end), len(begin), new_job)
        ts_count += 1

        new_end = []
        for i in range(0, len(begin)):
            if begin[i] == 1 and np.sum(d_matrix[i]) == 0:
                if end == 0:
                    new_end.append(i)
                end = 1

        if sum(end) == len(job_arrive_time):
            return 1

        # 消除旧业务占用+放置新业务
        reserve_server, inc_reserve = job_exit(rack_num, local_solution, reserve_server, new_end, inc_usage, inc_reserve)
        for i in new_end:
            end[i] = 1
        local_solution, reserve_server, d_matrix = job_deploy(local_solution, reserve_server, new_job, job_worker_num,
                                                              d_matrix, d_per_worker)
        # 分inc资源
        inc_benefit = benefit_cal(new_job, local_solution, inc_benefit, rack_num, d_per_worker)
        inc_usage, inc_reserve, inc_benefit, inc_job = inc_allocate(inc_reserve, inc_benefit, inc_job, job_wait,
                                                                    rack_num, inc_usage)

        # 获取新数据矩阵
        d_matrix = data_obtain(inc_usage, rack_num, d_per_worker, d_matrix, new_job, local_solution)

        b_per_worker_old = copy.deepcopy(b_per_worker)
        # 分带宽
        b_per_worker, begin, b_inter_allocate = bandwidth_allocate(d_matrix, job_agg, local_solution, algo, inc_usage,
                                                                   rack_num, b_tor, b_oxc_port, port_per_rack, b_unit,
                                                                   b_per_worker, begin)

        # 判断重配带宽
        b_recon = recon(oxc_topo, b_inter_allocate, rack_num, b_oxc_port, b_per_worker_old, b_per_worker, inc_usage,
                        local_solution, begin, end)

        # 更新剩余数据量
        d_matrix = communication(d_matrix, inc_usage, b_per_worker, b_recon, local_solution, rack_num, t_recon)


arrive_time = []
worker_num = []
with open("simulate_time.txt", 'r') as file:
    for line in file:
        # 去除行尾的换行符，并以逗号分割行数据
        columns = line.strip().split(",")
        time_job = float(columns[0])
        arrive_time.append(time_job)

with open("simulate_worker.txt", 'r') as file:
    for line in file:
        # 去除行尾的换行符，并以逗号分割行数据
        columns = line.strip().split(",")
        num = int(columns[0])
        worker_num.append(num)

schedule(8, 64, 60, arrive_time, worker_num, [], [], [], [], 1)
