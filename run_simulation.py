import sys
import numpy as np
import os
import new_algorithm

#t:时隙长数组索引;k：机架数数组索引; n：业务数数组索引; r：lambda数组索引; c：INC数组索引
def schedule_all (t, k, n, r, c):
    routine = "simulate_time" + str(lam[r]) + ".txt"
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
    
    #不同到达率的到达时间文件        
    with open(routine, 'r') as file:
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
    
    
    output = "output5_10_20" + ".txt"
    file1 = open(output, 'a')
            # 重定向标准输出到文件
    original_stdout = sys.stdout
    sys.stdout = file1
    print("lamda=", lam[r], "; number of job=", num_job[n], "; number of rack=", rack_num[k], "; length of Ts=", t_recon[t],
          "; INC=", inc_limit[c])

    #FCFS-noINC
    new_algorithm.schedule(rack_num[k], server_per_rack, 50, arrive_time[:50], worker_num[:50], agg_time[:50], d_worker[:50], 0, ts_len[t], 0, b_tor,
                b_oxc_port, port_num, b_unit, t_recon)
    #Algo-noINC
    new_algorithm.schedule(rack_num[k], server_per_rack, 50, arrive_time[:50], worker_num[:50], agg_time[:50], d_worker[:50], 1, ts_len[t], 0, b_tor,
                b_oxc_port, port_num, b_unit, t_recon)
    #FCFS-INC
    new_algorithm.schedule(rack_num[k], server_per_rack, 50, arrive_time[:50], worker_num[:50], agg_time[:50], d_worker[:50], 0, ts_len[t], inc_limit[c], b_tor,
                b_oxc_port, port_num, b_unit, t_recon)
    #Algo-INC
    new_algorithm.schedule(rack_num[k], server_per_rack, 50, arrive_time[:50], worker_num[:50], agg_time[:50], d_worker[:50], 1, ts_len[t], inc_limit[c], b_tor,
                b_oxc_port, port_num, b_unit, t_recon)

    sys.stdout = original_stdout
    file1.close()
    

# 定义变量
rack_num = [64,32,16,8]  # 机架数目
server_per_rack = 64 #每机架server数目
init_num = 50 #初始业务数目
# arrive_time = []
# worker_num = []
# agg_time = []
# d_worker = []
ts_len = [20,15,10,5,1]  # 时间片
inc_limit = [20, 15, 10, 5] #inc资源上限
b_tor = [480 for i1 in range(0, rack_num)] #每rack交换机带宽容量，B=480 or 240?
b_oxc_port = [40 for i2 in range(0, rack_num)] #每port带宽
port_num = 12 #port数目
b_unit = 1 #？
t_recon = 0.2  # oxc重构时间
lam = [20]
num_job = [800,1000,1200,1400]

# 默认场景：
#t:时隙长数组索引;k：机架数数组索引; n：业务数数组索引; r：lambda数组索引; c：INC数组索引
schedule_all(3, 0, -1, -1, 2) #5,64,1400,lam=20,inc=10

# 时隙长 #跳过默认时隙
for t1 in range(0, len(ts_len)):
    if ts_len[t1] == 5:
        continue
    schedule_all(t1, 0, 0, -1, 2)#t1,64,800,lam=20,inc=10

# 业务数 
for n1 in range(0, len(num_job)):
    if num_job[n1] == 1400:
        continue
    schedule_all(3, 0, n1, -1, 2)#5,64,n1,lam=20,inc=10

# lambda
for r1 in range(0, len(lam)):
    if lam[r1] == 1000:
        continue
    schedule_all(3, 0, 0, r1, 2)#5,64,800,r1,inc=10

# INC
for c1 in range(0, len(inc_limit)):
    if inc_limit[c1] == 10:
        continue
    schedule_all(3, 0, -1, -1, c1)#5,64,800,lam=20,c1
    
# 机架数
for k1 in range(0, len(rack_num)):
    if rack_num[k1] == 64:
        continue
    schedule_all(3, k1, 0, -1, 2)#5, k1, 800, lam=20, 10
