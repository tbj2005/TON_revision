import numpy as np
import copy
from scipy.optimize import linear_sum_assignment
import time


def job_exit(rack_num, local_solution, reserve_server, add_end, inc_usage, inc_reserve):
    for i in add_end:
        worker = np.sum(local_solution[i], axis=1)
        ps = np.sum(local_solution[i], axis=0)
        for r in range(0, rack_num):
            reserve_server[r] += worker[r]
            if ps[r] > 0:
                reserve_server[r] += 1
            if inc_usage[r][i] == 1:
                inc_reserve[r] += 1
    return reserve_server, inc_reserve


def job_deploy(local_solution, reserve_server, new_job, job_worker_num):
    pending_job = []
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
        else:
            pending_job.append(i)
    return local_solution, reserve_server, pending_job


def benefit_cal(new_job, local_solution, inc_benefit, rack_num, d_per_worker):
    for i in new_job:
        ps = np.sum(local_solution[i], axis=0)
        ps_local = np.argmax(ps)
        if ps[ps_local] == 0:
            continue
        worker = np.sum(local_solution[i], axis=1)
        if np.count_nonzero(worker) == 1:
            inc_benefit[np.argmax(worker)][i] += 1
            if worker[ps_local] > 0:
                for j in range(0, rack_num):
                    inc_benefit[j][i] += 0
            else:
                for j in range(0, rack_num):
                    inc_benefit[j][i] += local_solution[i][j][ps_local] * d_per_worker[i]
        else:
            for j in range(0, rack_num):
                if local_solution[i][j][ps_local] > 0:
                    inc_benefit[j][i] += (local_solution[i][j][ps_local] - 1) * d_per_worker[i]
    return inc_benefit


def inc_allocate(inc_reserve, inc_benefit, inc_job, job_wait, rack_num, inc_usage):
    inc_benefit_copy = copy.deepcopy(inc_benefit)
    while np.max(inc_benefit_copy) > 0 and np.sum(inc_reserve) > 0:
        row = [i for i in range(0, rack_num) if inc_reserve[i] == 0]
        col = [i for i in range(0, len(inc_job)) if i not in job_wait]
        for i in row:
            inc_benefit_copy[i, :] = -1
        for i in col:
            inc_benefit_copy[:, i] = -1
        row_ch, col_ch = linear_sum_assignment(inc_benefit_copy, True)
        for i in range(0, len(row_ch)):
            if inc_benefit_copy[row_ch[i]][col_ch[i]] >= 0:
                inc_usage[row_ch[i]][col_ch[i]] = 1
                inc_reserve[row_ch[i]] -= 1
                inc_benefit_copy[:, col_ch[i]] = -1
                inc_benefit[:, col_ch[i]] = -1
                inc_job[col_ch[i]] = 1
    return inc_usage, inc_reserve, inc_benefit, inc_job


def data_obtain(inc_usage, rack_num, d_per_worker, d_matrix, new_job, local_solution):
    for i in new_job:
        ps = np.sum(local_solution[i], 0)
        ps_local = np.argmax(ps)
        worker = np.sum(local_solution[i], 1)
        if np.count_nonzero(worker) == 1:
            all_in = 1
        else:
            all_in = 0
        for j in range(0, rack_num):
            if inc_usage[j][i] == 0:
                d_matrix[i][j][ps_local] = local_solution[i][j][ps_local] * d_per_worker[i]
            elif all_in == 1:
                d_matrix[i][j][j] = local_solution[i][j][ps_local] * d_per_worker[i]
            elif all_in == 0:
                d_matrix[i][j][ps_local] = d_per_worker[i]
    return d_matrix


def pct_rack(inc_usage, d_matrix, rack_index, local_solution, rack_num, job_agg, b_tor, b_per_port, port_per_rack):
    relate_job = []
    traffic = []
    agg = []
    relate_job_oxc = []
    traffic_oxc = []
    agg_oxc = []
    for i in range(0, len(d_matrix)):
        if np.sum(d_matrix[i]) == 0:
            continue
        ps = np.sum(local_solution[i], axis=0)
        ps_local = np.argmax(ps)
        worker = np.sum(local_solution[i], axis=1)
        if np.count_nonzero(worker) == 1:
            all_in = 1
        else:
            all_in = 0
        if worker[rack_index] > 0:
            relate_job.append(i)
            if inc_usage[rack_index][i] == 1:
                if all_in == 1:
                    traffic.append(d_matrix[i][rack_index][rack_index])
                if all_in == 0:
                    traffic.append(d_matrix[i][rack_index][ps_local] * local_solution[i][rack_index][ps_local])
            elif inc_usage[rack_index][i] == 0:
                traffic.append(d_matrix[i][rack_index][ps_local])
            if np.count_nonzero(local_solution[i]) > 1:
                agg.append(job_agg[i])
            elif sum([inc_usage[j][i] for j in range(0, rack_num)]) == 0:
                agg.append(job_agg[i])
            else:
                agg.append(0)
        elif ps_local == rack_index:
            if sum([inc_usage[v][i] for v in range(0, rack_num)]) * all_in == 0:
                relate_job.append(i)
                traffic.append(sum([d_matrix[i][j][rack_index] for j in range(0, rack_num)]))
                if np.count_nonzero(local_solution[i]) > 1:
                    agg.append(job_agg[i])
                elif sum([inc_usage[j][i] for j in range(0, rack_num)]) == 0:
                    agg.append(job_agg[i])
                else:
                    agg.append(0)

        if sum([d_matrix[i][rack_index][j] for j in range(0, rack_num) if j != rack_index]) > 0:
            relate_job_oxc.append(i)
            traffic_oxc.append(sum([d_matrix[i][rack_index][j] for j in range(0, rack_num) if j != rack_index]))
            if np.count_nonzero(local_solution[i]) > 1:
                agg_oxc.append(job_agg[i])
            elif sum([inc_usage[j][i] for j in range(0, rack_num)]) == 0:
                agg_oxc.append(job_agg[i])
            else:
                agg_oxc.append(0)
        if sum([d_matrix[i][j][rack_index] for j in range(0, rack_num) if j != rack_index]) > 0:
            relate_job_oxc.append(i)
            traffic_oxc.append(sum([d_matrix[i][j][rack_index] for j in range(0, rack_num) if j != rack_index]))
            if np.count_nonzero(local_solution[i]) > 1:
                agg_oxc.append(job_agg[i])
            elif sum([inc_usage[j][i] for j in range(0, rack_num)]) == 0:
                agg_oxc.append(job_agg[i])
            else:
                agg_oxc.append(0)

    # 上行：
    threshold = 0.01
    pct = bisection(relate_job, agg, traffic, b_tor[rack_index], threshold)
    pct_oxc = bisection(relate_job_oxc, agg_oxc, traffic_oxc, b_per_port[rack_index] * port_per_rack, threshold)
    return pct, pct_oxc, relate_job, traffic, agg, relate_job_oxc, traffic_oxc, agg_oxc


def bisection(relate, agg, traffic, bandwidth, threshold):
    t_max_up = max(agg + [0]) + sum(traffic) / bandwidth
    t_min_up = max(sum(traffic) / bandwidth, max(agg + [0]))
    if sum(agg) == 0:
        return sum(traffic) / bandwidth
    if len(relate) == 1:
        return max(agg + [0]) + sum(traffic) / bandwidth
    while t_max_up - t_min_up >= threshold:
        t_mid_up = (t_max_up + t_min_up) / 2
        if sum([traffic[i] / (t_mid_up - agg[i]) for i in range(0, len(relate))]) < bandwidth:
            t_max_up = t_mid_up
        else:
            t_min_up = t_mid_up
    return t_min_up


def pct_count(inc_usage, d_matrix, rack_num, local_solution, job_agg, b_tor, b_oxc_port, port_per_rack):
    port_t = np.array([0 for i in range(0, 2 * rack_num)], dtype=float)
    relate = []
    relate_oxc = []
    tra = []
    tra_oxc = []
    a = []
    a_oxc = []
    for i in range(0, rack_num):
        pct, pct_oxc, relate_job, traffic, agg, relate_job_oxc, traffic_oxc, agg_oxc = (
            pct_rack(inc_usage, d_matrix, i, local_solution, rack_num, job_agg, b_tor, b_oxc_port, port_per_rack))
        port_t[i] += pct
        port_t[i + rack_num] += pct_oxc
        relate.append(relate_job)
        relate_oxc.append(relate_job_oxc)
        tra.append(traffic)
        tra_oxc.append(traffic_oxc)
        a.append(agg)
        a_oxc.append(agg_oxc)
    return port_t, relate, tra, a, relate_oxc, tra_oxc, a_oxc


def feasible_add(b_per_worker, relate, local_solution, b_tor_bi, b_inter_bi, rack_num, inc_usage, b_tor,
                 b_oxc_port, port_per_rack, agg_time, b_unit, d_matrix):
    t_add = []
    for job_index in relate:
        if np.sum(d_matrix[job_index]) == 0:
            t_add.append(-1)
            continue
        ps = np.sum(local_solution[job_index], axis=0)
        ps_local = np.argmax(ps)
        worker = np.sum(local_solution[job_index], axis=1)
        if np.count_nonzero(worker) == 1:
            all_in = 1
        else:
            all_in = 0
        b_tor_bi_test = copy.deepcopy(b_tor_bi)
        b_inter_bi_test = copy.deepcopy(b_inter_bi)
        p_inter_test = np.zeros([rack_num, rack_num])
        b_tor_bi_test[ps_local] += (b_unit * sum([(inc_usage[v][job_index] * (1 - all_in) +
                                                   (1 - inc_usage[v][job_index]) *
                                                   local_solution[job_index][v][ps_local]) for v in
                                                  range(0, rack_num)]))
        for u in range(0, rack_num):
            b_tor_bi_test[u] += local_solution[job_index][u][ps_local] * b_unit

            if u != ps_local:
                b_inter_bi_test[u][ps_local] += (inc_usage[u][job_index] * (1 - all_in) + (1 - inc_usage[u][job_index])
                                                 * local_solution[job_index][u][ps_local]) * b_unit
                b_inter_bi_test[ps_local][u] += (inc_usage[u][job_index] * (1 - all_in) + (1 - inc_usage[u][job_index])
                                                 * local_solution[job_index][u][ps_local]) * b_unit
        for u in range(0, rack_num):
            for v in range(0, rack_num):
                if b_inter_bi_test[u][v] > 0:
                    p_inter_test[u][v] = np.ceil(b_inter_bi_test[u][v] / b_oxc_port[v])
        in_resource = np.array([b_tor[i] - b_tor_bi_test[i] for i in range(0, rack_num)])
        p_inter_up = np.sum(p_inter_test, axis=1)
        if min(in_resource) < 0:
            t_add.append(-1)
        elif max(p_inter_up) > port_per_rack:
            t_add.append(-1)
        else:
            t_matrix = np.zeros([rack_num, rack_num])
            for i in range(0, rack_num):
                for j in range(0, rack_num):
                    if d_matrix[job_index][i][j] > 0 and i != j:
                        t_matrix[i][j] = (d_matrix[job_index][i][j] / (inc_usage[i][job_index] +
                                                                       (1 - inc_usage[i][job_index]) *
                                                                       local_solution[job_index][i][j] *
                                                                       (b_per_worker[job_index] + b_unit)) + agg_time[
                                              relate.index(job_index)])
                    if d_matrix[job_index][i][j] > 0 and i == j:
                        t_matrix[i][j] = d_matrix[job_index][i][j] / (local_solution[job_index][i][ps_local] *
                                                                      (b_per_worker[job_index] + b_unit))
            t_add.append(np.max(t_matrix))
    t_add = np.array([i for i in t_add])
    return t_add


def bandwidth_allocate(d_matrix, job_agg, local_solution, algo, inc_usage, rack_num, b_tor, b_oxc_port, port_per_rack,
                       b_unit, b_per_worker, begin):
    b_tor_up = np.zeros(rack_num)
    b_tor_down = np.zeros(rack_num)
    b_inter = np.zeros([rack_num, rack_num])
    for i in range(0, len(d_matrix)):
        if np.sum(d_matrix[i]) == 0:
            b_per_worker[i] = 0
        if np.sum(d_matrix[i]) > 0 and begin[i] == 1:
            # b_per_worker[i] = b_unit
            ps = np.sum(local_solution[i], axis=0)
            ps_local = np.argmax(ps)
            worker = np.sum(local_solution[i], axis=1)
            if np.count_nonzero(worker) == 1:
                all_in = 1
            else:
                all_in = 0
            for u in range(0, rack_num):
                b_tor_up[u] += local_solution[i][u][ps_local] * b_per_worker[i]
                if u != ps_local:
                    b_inter[u][ps_local] += (inc_usage[u][i] * (1 - all_in) + (1 - inc_usage[u][i]) *
                                             local_solution[i][u][ps_local]) * b_per_worker[i]
            b_tor_down[ps_local] += (b_per_worker[i] * sum([(inc_usage[v][i] * (1 - all_in) + (1 - inc_usage[v][i]) *
                                                             local_solution[i][v][ps_local]) for v in
                                                            range(0, rack_num)]))

    b_tor_bi = b_tor_up + b_tor_down
    b_inter_bi = b_inter + b_inter.T
    if algo == 1:
        port_t, relate, tra, agg, relate_oxc, tra_oxc, agg_oxc = (
            pct_count(inc_usage, d_matrix, rack_num, local_solution, job_agg, b_tor, b_oxc_port, port_per_rack))
        sort_port = np.argsort(port_t)[::-1]

        job_allocated = []
        for i in sort_port:
            if port_t[i] == 0:
                break
            if i < rack_num:
                relate_j = relate[i]
                agg_t = agg[i]
                agg_t = [agg_t[k] for k in range(0, len(agg_t)) if relate_j[k] not in job_allocated]
                relate_j = [k for k in relate_j if k not in job_allocated]
            elif i < 2 * rack_num:
                relate_j = relate_oxc[i - rack_num]
                agg_t = agg_oxc[i - rack_num]
                agg_t = [agg_t[k] for k in range(0, len(agg_t)) if relate_j[k] not in job_allocated]
                relate_j = [k for k in relate_j if k not in job_allocated]
            while 1:
                t_add = feasible_add(b_per_worker, relate_j, local_solution, b_tor_bi, b_inter_bi,
                                     rack_num, inc_usage, b_tor, b_oxc_port, port_per_rack, agg_t, b_unit,
                                     d_matrix)
                if len(t_add) == 0:
                    break
                max_index = np.argmax(t_add)
                if t_add[max_index] != -1:
                    b_per_worker[relate_j[max_index]] += b_unit
                    begin[relate_j[max_index]] = 1
                    job_index = relate_j[max_index]
                    ps = np.sum(local_solution[job_index], axis=0)
                    ps_local = np.argmax(ps)
                    worker = np.sum(local_solution[job_index], axis=1)
                    if np.count_nonzero(worker) == 1:
                        all_in = 1
                    else:
                        all_in = 0
                    for u in range(0, rack_num):
                        b_tor_bi[u] += local_solution[job_index][u][ps_local] * b_unit
                        if u != ps_local:
                            b_inter_bi[u][ps_local] += (inc_usage[u][job_index] * (1 - all_in) +
                                                        (1 - inc_usage[u][job_index]) *
                                                        local_solution[job_index][u][ps_local]) * b_unit
                            b_inter_bi[ps_local][u] += (inc_usage[u][job_index] * (1 - all_in) +
                                                        (1 - inc_usage[u][job_index]) *
                                                        local_solution[job_index][u][ps_local]) * b_unit
                    b_tor_bi[ps_local] += (b_unit * sum([(inc_usage[v][job_index] * (1 - all_in) +
                                                          (1 - inc_usage[v][job_index]) * local_solution[job_index][v][
                                                              ps_local]) for
                                                         v in range(0, rack_num)]))
                else:
                    for k in relate_j:
                        job_allocated.append(k)
                    break

        return b_per_worker, begin, b_inter_bi
    if algo == 0:
        for job_index in range(0, len(d_matrix)):
            flag = 0
            if np.sum(d_matrix[job_index]) > 0:
                ps = np.sum(local_solution[job_index], axis=0)
                ps_local = np.argmax(ps)
                worker = np.sum(local_solution[job_index], axis=1)
                if np.count_nonzero(worker) == 1:
                    all_in = 1
                else:
                    all_in = 0
                while 1:
                    b_tor_bi_test = copy.deepcopy(b_tor_bi)
                    b_inter_bi_test = copy.deepcopy(b_inter_bi)
                    p_inter_test = np.zeros([rack_num, rack_num])
                    b_tor_bi_test[ps_local] += (b_unit * sum([(inc_usage[v][job_index] * (1 - all_in) +
                                                               (1 - inc_usage[v][job_index]) *
                                                               local_solution[job_index][v][ps_local]) for v in
                                                              range(0, rack_num)]))
                    for u in range(0, rack_num):
                        b_tor_bi_test[u] += local_solution[job_index][u][ps_local] * b_unit

                        if u != ps_local:
                            b_inter_bi_test[u][ps_local] += (inc_usage[u][job_index] * (1 - all_in) + (
                                        1 - inc_usage[u][job_index])
                                                             * local_solution[job_index][u][ps_local]) * b_unit
                            b_inter_bi_test[ps_local][u] += (inc_usage[u][job_index] * (1 - all_in) + (
                                        1 - inc_usage[u][job_index])
                                                             * local_solution[job_index][u][ps_local]) * b_unit
                    for u in range(0, rack_num):
                        for v in range(0, rack_num):
                            if b_inter_bi_test[u][v] > 0:
                                p_inter_test[u][v] = np.ceil(b_inter_bi_test[u][v] / b_oxc_port[v])
                    in_resource = np.array([b_tor[i] - b_tor_bi_test[i] for i in range(0, rack_num)])
                    p_inter_up = np.sum(p_inter_test, axis=1)

                    if min(in_resource) >= 0 and max(p_inter_up) <= port_per_rack:
                        b_per_worker[job_index] += b_unit
                        b_tor_bi = copy.deepcopy(b_tor_bi_test)
                        b_inter_bi = copy.deepcopy(b_inter_bi_test)
                        begin[job_index] = 1
                        flag = 1
                    else:
                        break
                if flag == 0:
                    break

    return b_per_worker, begin, b_inter_bi


def recon(oxc_topo, b_inter, rack_num, b_oxc_port, b_per_worker_old, b_per_worker, inc_usage, local_solution, begin,
          end):
    new_oxc_topo = np.zeros([rack_num, rack_num])
    recon_bandwidth = np.zeros(len(b_per_worker))
    for i in range(0, rack_num):
        for j in range(0, rack_num):
            if i > j:
                new_oxc_topo[i][j] = np.ceil(b_inter[i][j] / b_oxc_port[i])
                new_oxc_topo[j][i] = np.ceil(b_inter[j][i] / b_oxc_port[i])
    add_oxc_topo = new_oxc_topo - oxc_topo
    for i in range(0, rack_num):
        for j in range(0, rack_num):
            for k in range(0, len(b_per_worker)):
                if add_oxc_topo[i][j] > 0 and begin[k] == 1 and end[k] == 0 and local_solution[k][i][j] > 0:
                    if np.count_nonzero(local_solution[k]) > 1:
                        recon_bandwidth[k] = max(b_per_worker[k] - b_per_worker_old[k], 0)
                    elif inc_usage[i][k] == 0:
                        recon_bandwidth[k] = max(b_per_worker[k] - b_per_worker_old[k], 0)
    oxc_topo = copy.deepcopy(new_oxc_topo)
    return recon_bandwidth, oxc_topo


def communication(d_matrix, inc_usage, b_per_worker, recon_bandwidth, local_solution, rack_num, t_recon, ts_len):
    for i in range(0, len(d_matrix)):
        if b_per_worker[i] > 0:
            ps = np.sum(local_solution[i], axis=0)
            ps_local = np.argmax(ps)
            for u in range(0, rack_num):
                for v in range(0, rack_num):
                    if d_matrix[i][u][v] > 0:
                        if u != v:
                            data_tran = min(b_per_worker[i] * (local_solution[i][u][v] * (1 - inc_usage[u][i]) +
                                            inc_usage[u][i]) * ts_len - local_solution[i][u][v] * recon_bandwidth[i] *
                                            t_recon, d_matrix[i][u][v])
                        else:
                            data_tran = min(b_per_worker[i] * local_solution[i][u][ps_local], d_matrix[i][u][v])
                        d_matrix[i][u][v] -= data_tran
    return d_matrix


def schedule(rack_num, server_per_rack, init_num, job_arrive_time, job_worker_num, job_agg, d_per_worker, algo, ts_len,
             inc_limit, b_tor, b_oxc_port, port_per_rack, b_unit, t_recon):
    job_list = []
    ts_count = 0
    recon_count = 0
    t_job = np.zeros(len(job_arrive_time))
    local_solution = []
    reserve_server = np.array([server_per_rack for i in range(0, rack_num)])
    inc_usage = np.zeros([rack_num, len(job_worker_num)])
    inc_reserve = np.array([inc_limit for i in range(0, rack_num)])
    inc_benefit = -1 * np.ones([rack_num, len(job_worker_num)])
    inc_job = []
    end = []
    begin = []
    d_matrix = []
    b_per_worker = []
    job_wait = []
    oxc_topo = np.zeros([rack_num, rack_num])
    pend_job = []
    while 1:
        if ts_count == 0:
            # 首个时隙信息更新
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
            if count_job < len(job_arrive_time):
                if init_num != 0:
                    while job_arrive_time[count_job] <= ts_count * ts_len + job_arrive_time[init_num - 1]:
                        new_job.append(count_job)
                        job_list.append(count_job)
                        job_wait.append(count_job)
                        inc_job.append(0)
                        local_solution.append(np.zeros([rack_num, rack_num]))
                        d_matrix.append(np.zeros([rack_num, rack_num]))
                        b_per_worker.append(0)
                        end.append(0)
                        begin.append(0)
                        count_job += 1
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

        ts_count += 1

        new_end = []
        for i in range(0, len(begin)):
            if begin[i] == 1 and np.sum(d_matrix[i]) == 0:
                if end[i] == 0:
                    if sum(inc_usage[u][i] for u in range(0, rack_num)) == 1 and np.count_nonzero(local_solution[i]) == 1:
                        t_job[i] = ts_count * ts_len
                    else:
                        t_job[i] += job_agg[i] + ts_count * ts_len #agg_time[i] to job_agg[i]
                    new_end.append(i)
                    end[i] = 1

        if sum(end) == len(job_arrive_time):
            return np.max(t_job), recon_count

        new_job += pend_job
        # 消除旧业务占用+放置新业务
        reserve_server, inc_reserve = job_exit(rack_num, local_solution, reserve_server, new_end, inc_usage,
                                               inc_reserve)

        local_solution, reserve_server, pend_job = job_deploy(local_solution, reserve_server, new_job, job_worker_num)
        # 分inc资源
        if algo == 1:
            inc_benefit = benefit_cal(new_job, local_solution, inc_benefit, rack_num, d_per_worker)
            inc_usage, inc_reserve, inc_benefit, inc_job = inc_allocate(inc_reserve, inc_benefit, inc_job, job_wait,
                                                                        rack_num, inc_usage)
        if algo == 0:
            for i in job_list:
                if begin[i] == 0:
                    worker = np.sum(local_solution[i], axis=1)
                    for j in range(0, len(worker)):
                        if inc_reserve[j] == 0:
                            worker[j] = 0
                    inc_index = np.argmax(worker)
                    if worker[inc_index] == 0:
                        break
                    else:
                        inc_usage[inc_index][i] = 1
                        inc_reserve[inc_index] -= 1

        # 获取新数据矩阵
        d_matrix = data_obtain(inc_usage, rack_num, d_per_worker, d_matrix, new_job, local_solution)

        b_per_worker_old = copy.deepcopy(b_per_worker)

        # 分带宽
        b_per_worker, begin, b_inter_allocate = bandwidth_allocate(d_matrix, job_agg, local_solution, algo, inc_usage,
                                                                   rack_num, b_tor, b_oxc_port, port_per_rack, b_unit,
                                                                   b_per_worker, begin)
        job_wait = [ele for ele in job_wait if begin[ele] == 0]

        # 判断重配带宽
        b_recon, oxc_topo = recon(oxc_topo, b_inter_allocate, rack_num, b_oxc_port, b_per_worker_old, b_per_worker, inc_usage,
                                  local_solution, begin, end)
        if sum(b_recon) > 0:
            recon_count += 1
        # 更新剩余数据量
        d_matrix = communication(d_matrix, inc_usage, b_per_worker, b_recon, local_solution, rack_num, t_recon, ts_len)


# arrive_time = []
# worker_num = []
# agg_time = []
# d_worker = []
# rack_number = 4
# port_num = 12
# b_tor = [240 for i1 in range(0, rack_number)]
# b_oxc_port = [40 for i2 in range(0, rack_number)]
# with open("simulate_time.txt", 'r') as file:
#     for line in file:
#         # 去除行尾的换行符，并以逗号分割行数据
#         columns = line.strip().split(",")
#         time_job = float(columns[0])
#         arrive_time.append(time_job)

# with open("simulate_worker.txt", 'r') as file:
#     for line in file:
#         # 去除行尾的换行符，并以逗号分割行数据
#         columns = line.strip().split(",")
#         num = int(columns[0])
#         worker_num.append(num)

# with open("PS_time.txt", 'r') as file:
#     for line in file:
#         # 去除行尾的换行符，并以逗号分割行数据
#         columns = line.strip().split(",")
#         num = float(columns[0])
#         agg_time.append(num)

# with open("Datasize.txt", 'r') as file:
#     for line in file:
#         # 去除行尾的换行符，并以逗号分割行数据
#         columns = line.strip().split(",")
#         num = float(columns[0])
#         d_worker.append(num)


# t1, r1 = schedule(rack_number, 64, 150, arrive_time[:150], worker_num[:150], agg_time[:150], d_worker[:150], 0, 1, 0, b_tor,
#                 b_oxc_port, port_num, 1, 0.2)
# print("noINC-FCFS:",t1, r1)

# t2, r2 = schedule(rack_number, 64, 150, arrive_time[:150], worker_num[:150], agg_time[:150], d_worker[:150], 1, 1, 0, b_tor,
#                 b_oxc_port, port_num, 1, 0.2)
# print("noINC-Algo:",t2, r2)

# start_time1=time.time()
# t3, r3 = schedule(rack_number, 64, 150, arrive_time[:150], worker_num[:150], agg_time[:150], d_worker[:150], 0, 1, 1, b_tor,
#                 b_oxc_port, port_num, 1, 0.2)
# print("INC-FCFS:",t3, r3)
# end_time1=time.time()
# print("span_time:",end_time1-start_time1)

# start_time2=time.time()
# t4, r4 = schedule(rack_number, 64, 150, arrive_time[:150], worker_num[:150], agg_time[:150], d_worker[:150], 1, 1, 1, b_tor,
#                 b_oxc_port, port_num, 1, 0.2)
# print("INC-Algo:",t4, r4)
# end_time2=time.time()
# print("span_time:",end_time2-start_time2)
