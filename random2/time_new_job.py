# -*- coding: utf-8 -*-
import numpy as np
import random


def poisson_process(lam, num):
    """
    生成泊松过程产生的业务时间点
    :param lam: 泊松过程的速率参数
    :param T: 模拟的时间范围
    :return: 业务时间点列表
    """
    t = 0
    events_x = []
    while len(events_x) < num:
        dt = np.random.exponential(scale=1 / lam)
        t += dt
        if len(events_x) < num:
            events_x.append(t)
    return events_x


# 设置泊松过程的速率参数和模拟的时间范围
# lam = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]  # 泊松过程的速率参数，表示单位时间内平均产生lam个事件
lam = [20]
num = 1600 # 模拟的时间范围，单位为时间单位

# 生成业务时间点
events = []
for i in range(0, len(lam)):
    events.append(poisson_process(lam[i], num))

choices = [2, 4, 8, 16]
probabilities = [1 / len(choices) for i in range(0, len(choices))]

with open("simulate_worker.txt", "w") as file:
    file.write("")

for i in range(0, num):
    result = random.choices(choices, probabilities)[0]
    with open("simulate_worker.txt", 'a') as file:
        file.write(str(result) + '\n')

# 打印业务时间点
print("Poisson Process Events:")
for i in range(0, len(lam)):
    routine = "simulate_time" + ".txt"
    with open(routine, "w") as file:
        file.write("")
    count = 0
    while count < num:
        with open(routine, "a") as file:
            file.write(str(events[i][count]) + '\n')
        count += 1

with open("Datasize.txt", "w") as file:
    file.write("")

for i in range(0, num):
    result = random.choices([60.6, 107, 146, 308, 328, 344], [1/6 for i1 in range(0, 6)])[0]
    with open("Datasize.txt", "a") as file:
        file.write(str(result) + '\n')

with open("PS_time.txt", "w") as file:
    file.write("")

for i in range(1, num + 1):
    with open("PS_time.txt", "a") as file:
        D = 0
        W = 0
        with open("Datasize.txt", 'r') as file1:
            for line_number, line in enumerate(file1, 1):
                if line_number == i:
                    line = line.strip().split(",")[0]
                    D = float(line)
        with open("simulate_worker.txt", "r") as file2:
            for line_number, line in enumerate(file2, 1):
                if line_number == i:
                    line = line.strip().split(",")[0]
                    W = int(line)
        # print(D, W)
        if D == 60.6:
            T = 1.5 * W
        elif D == 107:
            T = 2.6 * W
        elif D == 146:
            T = 3.6 * W
        elif D == 308:
            T = 7.7 * W
        elif D == 328:
            T = 8.2 * W
        else:
            T = 8.6 * W
        file.write(str(T) + '\n')