import numpy as np
import gurobipy as gp
import heuristics_simulate
from gurobipy import *

m = 0.005
M = 1000

model = gp.Model("Time-sharing scheduling of jobs")
model.update()


def cal_i(ts):
    """
    该函数用于通过时间片数目获得连续时间子集，也就是集合I
    :param ts:number of time-shots
    :return:I
    """
    iter_ts = []
    for n in range(0, ts + 1):  # 持续时间含有多少个ts
        for i in range(1, ts + 1 - n):  # 持续时间从什么时候开始
            i1 = tuple([x for x in range(i, i + n + 1)])
            iter_ts.append(i1)
    return iter_ts


def ilp(job_num, ts_num, ts_len, rack_num, agg_time, n_inc, l_no_inc, b_in, b_out, d_job, b_port, p_oxc, t_recon,
        b_threshold):
    ts_set = cal_i(ts_num)
    t_all = model.addVar(lb=0, ub=GRB.INFINITY, obj=0.0, vtype=GRB.CONTINUOUS, name="t_all", column=None)
    t_j = model.addVars(job_num, vtype=GRB.CONTINUOUS, name="t_j")
    z_js = model.addVars(job_num, len(ts_set), vtype=GRB.BINARY, name="z_js")
    f_st = model.addVars(len(ts_set), ts_num, vtype=GRB.BINARY, name="f_st")
    k_ju = model.addVars(job_num, rack_num, vtype=GRB.BINARY, name="k_jv")
    k_j = model.addVars(job_num, vtype=GRB.BINARY, name="k_j")
    x_jt = model.addVars(job_num, ts_num, vtype=GRB.BINARY, name="x_jv")
    l_uj = model.addVars(rack_num, job_num, vtype=GRB.BINARY, name="l_vj")
    c_jt = model.addVars(job_num, ts_num, vtype=GRB.CONTINUOUS, name="c_ji")
    b_uvjt = model.addVars(rack_num, rack_num, job_num, ts_num, vtype=GRB.CONTINUOUS, name="b_uvji")
    d_uvjt = model.addVars(rack_num, rack_num, job_num, ts_num, vtype=GRB.CONTINUOUS, name="d_uvjt")
    p_uvt = model.addVars(rack_num, rack_num, ts_num, vtype=GRB.INTEGER, name="p_uvt")
    r_uvt = model.addVars(rack_num, rack_num, ts_num, vtype=GRB.BINARY, name="r_uvt")
    delta_b_uvjt = model.addVars(rack_num, rack_num, job_num, ts_num, vtype=GRB.CONTINUOUS, name="delta_b_uvjt")
    u_uvjt = model.addVars(rack_num, rack_num, job_num, ts_num, vtype=GRB.BINARY, name="u_uvjt")
    fn_uj = model.addVars(rack_num, job_num, vtype=GRB.INTEGER, name="flow_uj")
    fn_uvj = model.addVars(rack_num, rack_num, job_num, vtype=GRB.INTEGER, name="fn_uvj")
    ub_uvjt = model.addVars(rack_num, rack_num, job_num, ts_num, vtype=GRB.CONTINUOUS, name="ub_uvjt")

    w_uj = np.zeros([rack_num, job_num])
    ps_uj = np.zeros([rack_num, job_num])
    for j in range(0, job_num):
        worker = np.sum(l_no_inc[j], axis=1)
        ps = np.sum(l_no_inc[j], axis=0)
        for u in range(0, rack_num):
            w_uj[u][j] = worker[u]
            if ps[u] > 0:
                ps_uj[u][j] = 1
            else:
                ps_uj[u][j] = 0

    model.addConstrs(
        (x_jt[j, t] == quicksum(z_js[j, s] * f_st[s, t] for s in range(0, len(ts_set))) for j in range(0, job_num) for t
         in range(0, ts_num)), name="")

    model.addConstrs(
        (t_j[j] >= t * x_jt[j, t] + agg_time[j] * (1 - k_j[j]) for j in range(0, job_num) for t in range(0, ts_num)),
        name="")
    model.addConstrs((t_all >= t_j[j] for j in range(0, job_num)), name="")

    model.addConstrs((k_j[j] >= m * quicksum(k_ju[j, u] for u in range(0, rack_num)) for j in range(0, job_num)),
                     name="")
    model.addConstrs((k_j[j] <= M * quicksum(k_ju[j, u] for u in range(0, rack_num)) for j in range(0, job_num)),
                     name="")
    model.addConstrs((quicksum(k_ju[j, u] for u in range(0, rack_num)) <= 1 for j in range(0, job_num)), name="")

    model.addConstrs((quicksum(z_js[j, s] for s in range(0, len(ts_set))) == 1 for j in range(0, job_num)), name="")

    model.addConstrs((l_uj[u, j] <= M * w_uj[u][j] for j in range(0, job_num) for u in range(0, rack_num)), name="")
    model.addConstrs((l_uj[u, j] >= m * w_uj[u][j] for j in range(0, job_num) for u in range(0, rack_num)), name="")
    model.addConstrs((k_ju[j, u] <= M * w_uj[u][j] for j in range(0, job_num) for u in range(0, rack_num)), name="")
    model.addConstrs((quicksum(k_ju[j, u] * x_jt[j, t] for j in range(0, job_num)) <= n_inc[u] for u in
                      range(0, rack_num) for t in range(0, ts_num)), name="")

    model.addConstrs((c_jt[j, t] >= x_jt[j, t] * b_threshold for j in range(0, job_num) for t in range(0, ts_num)),
                     name="")
    model.addConstrs((c_jt[j, t] <= M * x_jt[j, t] for j in range(0, job_num) for t in range(0, ts_num)), name="")

    model.addConstrs((fn_uj[u, j] == k_ju[j, u] + (1 - k_ju[j, u]) * w_uj[u][j] for u in range(0, rack_num) for j in
                      range(0, job_num)), name="")
    for u in range(0, rack_num):
        for v in range(0, rack_num):
            if u != u:
                model.addConstrs((fn_uvj[u, v, j] == fn_uj[u, j] * ps_uj[v][j] for j in range(0, job_num)), name="")
            else:
                model.addConstrs((fn_uvj[u, u, j] == 0 for j in range(0, job_num)), name="")

    for u in range(0, rack_num):
        for v in range(0, rack_num):
            if u != v:
                model.addConstrs((b_uvjt[u, v, j, t] == fn_uvj[u, v, j] * c_jt[j, t] for j in range(0, job_num) for t in
                                  range(0, ts_num)), name="")
                model.addConstrs((d_uvjt[u, v, j, 0] == fn_uvj[u, v, j] * d_job[j] for j in range(0, job_num)), name="")
            if u == v:
                model.addConstrs((b_uvjt[u, u, j, t] == w_uj[u][j] * c_jt[j, t] for j in range(0, job_num) for t in
                                  range(0, ts_num)), name="")
                model.addConstrs((d_uvjt[u, u, j, 0] == w_uj[u][j] * d_job[j] for j in range(0, job_num)), name="")

    model.addConstrs(
        (b_in[u] >= quicksum(w_uj[u][j] * c_jt[j, t] for j in range(0, job_num)) for u in range(0, rack_num) for t in
         range(0, ts_num)), name="")

    model.addConstrs(
        (b_out[u] >= quicksum((1 - k_ju[j, u]) * sum([w_uj[v][j] for v in range(0, rack_num)]) * c_jt[j, t] for j in
                              range(0, job_num)) for u in range(0, rack_num) for t in range(0, ts_num)), name="")

    model.addConstrs((p_uvt[u, v, t] == p_uvt[v, u, t] for t in range(0, ts_num) for u in range(0, rack_num) for v in
                      range(0, rack_num)), name="")
    model.addConstrs((p_uvt[u, u, t] == 0 for u in range(0, rack_num) for t in range(0, ts_num)), name="")
    model.addConstrs(
        (p_oxc[u] >= quicksum(p_uvt[u, v, t] for v in range(0, rack_num)) for u in range(0, rack_num) for t in
         range(0, ts_num)), name="")
    for u in range(0, rack_num):
        for v in range(0, rack_num):
            if u != v:
                model.addConstrs(
                    (p_uvt[u, v, t] * b_port >= quicksum(x_jt[j, t] * b_uvjt[u, v, j, t] for j in range(0, job_num)) for
                     t in range(0, ts_num)), name="")
                model.addConstrs(
                    (p_uvt[u, v, t] * b_port >= quicksum(x_jt[j, t] * b_uvjt[v, u, j, t] for j in range(0, job_num)) for
                     t in range(0, ts_num)), name="")

    for u in range(0, rack_num):
        for v in range(0, rack_num):
            if u == v:
                model.addConstrs((r_uvt[u, u, t] == 0 for t in range(0, ts_num)), name="")
            else:
                model.addConstr(r_uvt[u, v, 0] * (p_uvt[u, v, 0] - 1) >= 0, name="")
                model.addConstr(r_uvt[u, v, 0] * M >= p_uvt[u, v, 0], name="")
                model.addConstrs((r_uvt[u, v, t] * (p_uvt[u, v, t] - 1) >= 0 for t in range(1, ts_num)), name="")
                model.addConstrs((r_uvt[u, v, t] * M >= p_uvt[u, v, t] for t in range(1, ts_num)), name="")

    for u in range(0, rack_num):
        for v in range(0, rack_num):
            model.addConstrs((delta_b_uvjt[u, v, j, 0] == b_uvjt[u, v, j, 0] for j in range(0, job_num)), name="")
            model.addConstrs(
                (delta_b_uvjt[u, v, j, t] == b_uvjt[u, v, j, t] - b_uvjt[u, v, j, t - 1] for t in range(1, ts_num) for j
                 in range(0, job_num)), name="")
            model.addConstrs((delta_b_uvjt[u, v, j, t] * u_uvjt[u, v, j, t] >= 0 for j in range(0, job_num) for t in
                              range(0, ts_num)), name="")
            model.addConstrs((M * u_uvjt[u, v, j, t] >= delta_b_uvjt[u, v, j, t] for j in range(0, job_num) for t in
                              range(0, ts_num)), name="")

    for u in range(0, rack_num):
        for v in range(0, rack_num):
            for j in range(0, job_num):
                model.addConstr(ub_uvjt[u, v, j, 0] == 0, name="")
                for t in range(1, ts_num):
                    model.addConstr(ub_uvjt[u, v, j, t] == delta_b_uvjt[u, v, j, t - 1] * u_uvjt[u, v, j, t - 1],
                                    name="")

    for u in range(0, rack_num):
        for v in range(0, rack_num):
            for j in range(0, job_num):
                model.addConstrs((d_uvjt[u, v, j, t] >= 0 for t in range(0, ts_num)), name="")
                model.addConstrs((d_uvjt[u, v, j, t] >= d_uvjt[u, v, j, t - 1] - x_jt[j, t - 1] * b_uvjt[
                    u, v, j, t - 1] * ts_len + ub_uvjt[u, v, j, t - 1] * r_uvt[u, v, t - 1] * t_recon for t in
                                  range(1, ts_num)), name="")

    model.addConstrs((d_uvjt[u, v, j, ts_num - 1] == 0 for u in range(0, rack_num) for v in range(0, rack_num) for j in
                      range(0, job_num)), name="")

    model.setObjective(t_all, GRB.MINIMIZE)

    model.setParam("OutputFlag", 1)
    model.Params.LogToConsole = True

    model.optimize()

    for v in model.getVars():
        (name, data) = (v.varName, v.x)
        if name[:5] == "t_all":
            print(name, data)


worker_job = []
Data_job = []
PS_time = []
with open("simulate_worker.txt", 'r') as file:
    for line in file:
        # 去除行尾的换行符，并以逗号分割行数据
        columns = line.strip().split(",")
        num_worker = int(columns[0])
        worker_job.append(num_worker)

with open("Datasize.txt", "r") as file:
    for line in file:
        columns = line.strip().split(",")
        Data = float(columns[0])
        Data_job.append(Data)

with open("PS_time.txt", "r") as file:
    for line in file:
        columns = line.strip().split(",")
        PS = float(columns[0])
        PS_time.append(PS)

job_number = 20
rack_number = 4
oxc_per_rack = 12
ts_number = 40
ts_length = 2
inc_lim = [2 for i in range(0, rack_number)]
b_in_rack = [40 * 12 for i in range(0, rack_number)]
b_out_rack = [40 * 12 for i in range(0, rack_number)]
oxc_port = [12 for i in range(0, rack_number)]
worker_job = np.array(worker_job[0:job_number])
Data_job = np.array(Data_job[0:job_number])
PS_time = np.array(PS_time[0:job_number])
w_t = heuristics_simulate.worker_local(rack_number, worker_job, begin=[0 for i in range(0, job_number)],
                                       end=[0 for i in range(0, job_number)],
                                       new_worker=[i for i in range(0, job_number)],
                                       W_t=[])
print(w_t)

ilp(job_number, ts_number, ts_length, rack_number, PS_time, inc_lim, w_t, b_in_rack, b_out_rack, Data_job, 40, oxc_port,
    0.3, 1)
