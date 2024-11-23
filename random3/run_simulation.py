import sys
import numpy as np
import os
import new_algorithm
# 定义变量
rack_num = [64,32,16,8]  # 机架数目
server_per_rack = 64 #每机架server数目
init_num = 500 #初始业务数目,越大算法性能越好
ts_len = [20,15,10,5]  # 时间片
inc_limit = [20, 15, 10, 5] #inc资源上限
b_tor = [240 for i1 in range(0, 64)] #每rack交换机带宽容量，B=480 or 240?
b_oxc_port = [40 for i2 in range(0, 64)] #每port带宽
port_num = 12 #port数目
b_unit = 1 #分配粒度
t_recon = 0.5  # oxc重构时间
lam = [20]
num_job = [800,1000,1200,1400]
#t:时隙长数组索引;k：机架数数组索引; n：业务数数组索引; r：lambda数组索引; c：INC数组索引
def schedule_all (t, k, n, c,init_num):
    # 定义变量
    arrive_time = []
    worker_num = []
    agg_time = []
    d_worker = []
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
    
    
    output = "output_random3" + ".txt"
    file1 = open(output, 'a')
            # 重定向标准输出到文件
    original_stdout = sys.stdout
    sys.stdout = file1
    print("number of job=", num_job[n], "; number of rack=", rack_num[k], "; length of Ts=", ts_len[t], "; INC=", inc_limit[c])

    #FCFS-noINC
    t1,r1 = new_algorithm.schedule(rack_num[k], server_per_rack, init_num, arrive_time[:num_job[n]], worker_num[:num_job[n]], agg_time[:num_job[n]], d_worker[:num_job[n]], 0, ts_len[t], 0, b_tor,
                b_oxc_port, port_num, b_unit, t_recon)
    print("noINC-FCFS:",t1, r1)
    #Algo-noINC
    t2,r2 = new_algorithm.schedule(rack_num[k], server_per_rack, init_num, arrive_time[:num_job[n]], worker_num[:num_job[n]], agg_time[:num_job[n]], d_worker[:num_job[n]], 1, ts_len[t], 0, b_tor,
                b_oxc_port, port_num, b_unit, t_recon)
    print("noINC-Algo:",t2, r2)
    #FCFS-INC
    t3,r3 = new_algorithm.schedule(rack_num[k], server_per_rack, init_num, arrive_time[:num_job[n]], worker_num[:num_job[n]], agg_time[:num_job[n]], d_worker[:num_job[n]], 0, ts_len[t], inc_limit[c], b_tor,
                b_oxc_port, port_num, b_unit, t_recon)
    print("INC-FCFS:",t3, r3)
    #Algo-INC
    t4,r4 = new_algorithm.schedule(rack_num[k], server_per_rack, init_num, arrive_time[:num_job[n]], worker_num[:num_job[n]], agg_time[:num_job[n]], d_worker[:num_job[n]], 1, ts_len[t], inc_limit[c], b_tor,
                b_oxc_port, port_num, b_unit, t_recon)
    print("INC-Algo:",t4, r4)

    sys.stdout = original_stdout
    file1.close()
    

# 默认场景：
#t:时隙长数组索引;k：机架数数组索引; n：业务数数组索引; r：lambda数组索引; c：INC数组索引
schedule_all(3, 0, -1, 2, init_num) #5,64,1400,,inc=10

# 业务数 
for n1 in range(0, len(num_job)):
    if num_job[n1] == 1400:
        continue
    schedule_all(3, 0, n1, 2, init_num)#5,64,n1,inc=10

# INC
for c1 in range(0, len(inc_limit)):
    if inc_limit[c1] == 10:
        continue
    schedule_all(3, 0, -1, c1, init_num)#5,64,1400,c1
    
# 机架数
for k1 in range(0, len(rack_num)):
    if rack_num[k1] == 64:
        continue
    schedule_all(3, k1, -1, 2, init_num)#5, k1, 1400, 10

# 时隙长 #跳过默认时隙
for t1 in range(0, len(ts_len)):
    if ts_len[t1] == 5:
        continue
    schedule_all(t1, 0, -1, 2, init_num)#t1,64,800,inc=10

