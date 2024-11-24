import sys
import numpy as np
import os
import new_algorithm
# 定义变量
ts_len = [20,15,10,5,1]  # 时间片t
rack_num = [64,32,16,8]  # 机架数目k
num_job = [1400,1200,1000,800] #业务数n
inc_limit = [20, 15, 10, 5] #inc资源上限c
init_num = 500 #初始业务数目

server_per_rack = 64 #每机架server数目
port_num = 64 #每机架oxc port数目，不能少于16保证连通性

b_unit = 10 #分配粒度
t_recon = 0.2  # oxc重构时间
lam = [20]

output_file = f"output_t.txt"

#t:时隙长数组索引;k：机架数数组索引; n：业务数数组索引; r：lambda数组索引; c：INC数组索引
def schedule_all (t, k, n, c,init_num):
    # 定义变量
    arrive_time = []
    worker_num = []
    agg_time = []
    d_worker = []
    b_oxc_port = [40 for i1 in range(0, rack_num[k])] #每oxc port带宽
    b_tor = [2560 for i2 in range(0, rack_num[k])] #每rack交换机带宽容量，B=64*40*2/2=2560

    with open("simulate_worker.txt", 'r') as file:
        for line in file:
            # 去除行尾的换行符，并以逗号分割行数据
            columns = line.strip().split(",")
            num = int(columns[0])
            worker_num.append(num)
          
    with open("simulate_time.txt", 'r') as file:
        for line in file:
            # 去除行尾的换行符，并以逗号分割行数据
            columns = line.strip().split(",")
            time_job = float(columns[0])
            arrive_time.append(time_job)
            
    with open("Datasize.txt", 'r') as file:
        for line in file:
            # 去除行尾的换行符，并以逗号分割行数据
            columns = line.strip().split(",")
            num = float(columns[0])
            d_worker.append(num)

    with open("PS_time.txt", 'r') as file:
        for line in file:
            # 去除行尾的换行符，并以逗号分割行数据
            columns = line.strip().split(",")
            num = float(columns[0])
            agg_time.append(num)

    # 写入文件
    with open(output_file, 'a') as file:
        file.write(f"length of Ts={ts_len[t]}; number of rack={rack_num[k]}; "
                   f"number of job={num_job[n]}; INC={inc_limit[c]}\n")

        # FCFS-noINC
        t1, r1 = new_algorithm.schedule(rack_num[k], server_per_rack, init_num,
                                        arrive_time[:num_job[n]], worker_num[:num_job[n]],
                                        agg_time[:num_job[n]], d_worker[:num_job[n]],
                                        0, ts_len[t], 0, b_tor, b_oxc_port, port_num, b_unit, t_recon)
        file.write(f"noINC-FCFS: {t1}, {r1}\n")

        # Algo-noINC
        t2, r2 = new_algorithm.schedule(rack_num[k], server_per_rack, init_num,
                                        arrive_time[:num_job[n]], worker_num[:num_job[n]],
                                        agg_time[:num_job[n]], d_worker[:num_job[n]],
                                        1, ts_len[t], 0, b_tor, b_oxc_port, port_num, b_unit, t_recon)
        file.write(f"noINC-Algo: {t2}, {r2}\n")

        # FCFS-INC
        t3, r3 = new_algorithm.schedule(rack_num[k], server_per_rack, init_num,
                                        arrive_time[:num_job[n]], worker_num[:num_job[n]],
                                        agg_time[:num_job[n]], d_worker[:num_job[n]],
                                        0, ts_len[t], inc_limit[c], b_tor, b_oxc_port, port_num, b_unit, t_recon)
        file.write(f"INC-FCFS: {t3}, {r3}\n")

        # Algo-INC
        t4, r4 = new_algorithm.schedule(rack_num[k], server_per_rack, init_num,
                                        arrive_time[:num_job[n]], worker_num[:num_job[n]],
                                        agg_time[:num_job[n]], d_worker[:num_job[n]],
                                        1, ts_len[t], inc_limit[c], b_tor, b_oxc_port, port_num, b_unit, t_recon)
        file.write(f"INC-Algo: {t4}, {r4}\n")
    

# 默认场景：
#t:时隙长数组索引;k：机架数数组索引; n：业务数数组索引; r：lambda数组索引; c：INC数组索引
print("default")
schedule_all(-1, 0, 2, 2, init_num) #ts=1, k=64, n=1000, c=10

# 业务数 
print("job's number")
for n1 in range(0, len(num_job)):
    if num_job[n1] == 1000:
        continue
    schedule_all(-1, 0, n1, 2, init_num)#5

# INC
print("inc's number")
for c1 in range(0, len(inc_limit)):
    if inc_limit[c1] == 10:
        continue
    schedule_all(-1, 0, 2, c1, init_num)#
    
# 机架数
print("rack's number")
for k1 in range(0, len(rack_num)):
    if rack_num[k1] == 64:
        continue
    schedule_all(-1, k1, 2, 2, init_num)#

# 时隙长 #跳过默认时隙
print("TS's length")
for t1 in range(0, len(ts_len)):
    if ts_len[t1] == 1:
        continue
    schedule_all(t1, 0, 2, 2, init_num)#

