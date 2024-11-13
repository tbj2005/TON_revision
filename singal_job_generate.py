import random
import numpy as np


class DataTransfer:
    def __init__(self, rack, data, ps, worker):
        self.rack = rack
        self.data = data
        self.ps = ps
        self.worker = worker

    def job_unload(self):
        """
        计算非卸载业务的数据传输矩阵
        :return: 返回非卸载业务的数据传输矩阵
        """
        local_ps = tuple([random.randint(1, self.rack) for x in range(0, self.ps)])
        local_worker = tuple([random.randint(1, self.rack) for x in range(0, self.worker)])
        DTM = np.zeros([self.rack, self.rack])
        V_T = np.zeros([self.rack, self.rack])
        for i in local_worker:
            for j in local_ps:
                DTM[i - 1][j - 1] += self.data
                V_T[i - 1][j - 1] += 1
        return DTM, V_T

    def job_load(self):
        """
        计算卸载业务的数据输入矩阵
        :return: 返回卸载业务的数据输入矩阵
        """
        local_worker = random.randint(1, self.rack)
        DIM = np.zeros(self.rack)
        V_I = np.zeros(self.rack)
        DIM[local_worker - 1] = self.data * self.worker
        V_I[local_worker - 1] = self.worker
        return DIM, V_I

    def job_test_no_INC(self):
        """
        计算非卸载业务的数据传输矩阵
        :return: 返回非卸载业务的数据传输矩阵
        """
        local_worker = tuple([random.randint(1, self.rack / 2) for x in range(0, self.worker)])
        local_ps = tuple([random.randint(self.rack / 2 + 1, self.rack) for x in range(0, self.ps)])
        DTM = np.zeros([self.rack, self.rack])
        V_T = np.zeros([self.rack, self.rack])
        for i in local_worker:
            for j in local_ps:
                DTM[i - 1][j - 1] += self.data
                V_T[i - 1][j - 1] += 1
        DIM = np.sum(DTM, axis=1)
        V_I = np.sum(V_T, axis=1)
        return DTM, V_T, DIM, V_I

    def job_test_INC(self):
        """
        计算卸载业务的数据输入矩阵
        :return: 返回卸载业务的数据输入矩阵
        """
        local = random.randint(1, self.rack / 2)
        local_worker = tuple([local for x in range(0, self.worker)])
        local_ps = tuple([random.randint(self.rack / 2 + 1, self.rack) for x in range(0, self.ps)])
        DTM = np.zeros([self.rack, self.rack])
        V_T = np.zeros([self.rack, self.rack])
        for i in local_worker:
            for j in local_ps:
                DTM[i - 1][j - 1] += self.data
                V_T[i - 1][j - 1] += 1
        DIM = np.sum(DTM, axis=1)
        V_I = np.sum(V_T, axis=1)
        return DTM, V_T, DIM, V_I
