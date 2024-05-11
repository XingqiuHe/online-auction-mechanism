import numpy as np
from Parameter import Parameter
import time
from numba import jit


@jit(nopython=True)
def cal_schedule(M, T, ti, tmi, si, wi, delta_t, Pi, h_U_i, sigma_2, p_w, p_s, p_f, C0, C1, C2, z0, z1, z2, fi):
    t_t_i = np.zeros((M, T))
    c_t_i = np.ones((M, T))*1e9
    t_c_i = np.zeros((M, T))
    c_c_i = np.ones((M, T))*1e9
    j_t_i = np.zeros(T, dtype=np.int64)
    j_c_i = np.zeros(T, dtype=np.int64)
    # 这里的上下限做过修改，与论文中有点不同
    # 枚举一个中间时刻作为t_p，用于分开计算传输和计算过程的开销
    target_fi = np.float(0.)
    max_tp = min(T, ti + tmi)
    for t_p in range(ti, max_tp):
        # 计算传输到计算站后，存在缓存中直到中间时刻t的开销
        for j in range(M):
            target_fi = si / delta_t / np.log2(1. + Pi * h_U_i[j] / sigma_2[j])
            # 先枚举开始向通信站传输的时间点t_t_0
            for t_t_0 in range(ti, t_p):    # t_p >= t_t_1 (t_t_1 = t_t_0 + 1)
                # 避免超出基站j在t时刻资源k的上限
                if C0[j] < z0[j][t_t_0] + target_fi:
                    continue
                isUnderCapacity = True

                temp = target_fi * p_w[j][t_t_0]
                # 再从t_t_0开始，计算存储在通信站的开销
                for tt in range(t_t_0, t_p):
                    if C2[j] < z2[j][tt] + si:
                        isUnderCapacity = False
                        break
                    temp += si * p_s[j][tt]

                if isUnderCapacity and temp < c_t_i[j][t_p]:
                    t_t_i[j][t_p] = t_t_0
                    c_t_i[j][t_p] = temp

        # 计算从中间时刻t_p开始，存在缓存中，直到完成计算的开销
        for j in range(M):
            # 枚举在计算站j里开始计算的时间点t_c_0
            for t_c_0 in range(t_p, max_tp):    # t_c_0 >= t_p
                # 避免超出基站j在t时刻资源k的上限
                if C1[j] < z1[j][t_c_0] + fi:
                    continue
                isUnderCapacity = True

                temp = fi * p_f[j][t_c_0]
                # 再计算从中间时刻t_p到开始计算时t_c_0的缓存开销
                for tt in range(t_p, t_c_0 + 1):
                    if C2[j] < z2[j][tt] + si:
                        isUnderCapacity = False
                        break
                    temp += si * p_s[j][tt]

                if isUnderCapacity and temp < c_c_i[j][t_p]:
                    t_c_i[j][t_p] = t_c_0
                    c_c_i[j][t_p] = temp
        c_t_i_t = c_t_i[:, t_p]
        c_c_i_t = c_c_i[:, t_p]
        # print(c_t_i_t, c_t_c_t)
        j_t_i[t_p] = np.argmin(c_t_i_t, axis=0)     # 保存以t_p作为站间传输时间的通信站
        j_c_i[t_p] = np.argmin(c_c_i_t, axis=0)     # 保存以t_p作为站间传输时间的计算站
        # 更新真正的fi (之前的fi为计算过程中的中间值)
        target_fi = si / delta_t / np.log2(1. + Pi * h_U_i[j_t_i[t_p]] / sigma_2[j_t_i[t_p]])

    temp = np.float(1e19)
    target_tp = np.int(-1)
    # print("j_t_i", j_t_i)
    # print("j_c_i", j_c_i)
    for t_p in range(ti, min(T, ti + tmi + 1)):
        c_t_i_t = c_t_i[:, t_p]
        c_c_i_t = c_c_i[:, t_p]
        # if temp > c_t_i[j_t_i[t_p]][t_p] + c_c_i[j_c_i[t_p]][t_p]:
        if temp > c_t_i_t[j_t_i[t_p]] + c_c_i_t[j_c_i[t_p]]:
            temp = c_t_i_t[j_t_i[t_p]] + c_c_i_t[j_c_i[t_p]]
            target_tp = t_p
    return target_fi, target_tp, t_t_i, c_t_i, t_c_i, c_c_i, j_t_i, j_c_i

class OnlineAlgorithm(object):
    def __init__(self, param: Parameter):
        super().__init__()
        self.param = param
        self.final_social_welfare = 0
        self.final_user_satisfaction = 0
        self.execute_time = 0

    '''执行在线算法'''
    def execute(self):
        start = time.time()

        # initial parameters
        for i in range(self.param.N):
            self.param.x[i] = 0
            for j in range(self.param.M):
                self.param.x_t[i][j] = 0
                self.param.x_c[i][j] = 0
        for j in range(self.param.M):
            for t in range(self.param.T):
                # k=0: w
                self.param.p_w[j][t] = self.param.L_w
                # k=1: f
                self.param.p_f[j][t] = self.param.L[1]
                # k=2: s
                self.param.p_s[j][t] = self.param.L[2]

        logs_socialwelfare = [0 for _ in range(self.param.N)]

        # 当任务i依次到达时
        for i in range(self.param.N):
            # 计算tao_max
            temp = 1
            while self.param.b[i](temp) >= 0:
                temp += 1
            self.param.tao_max[i] = temp - 1
            # 计算调度
            ret = self.schedulingAlgorithm_jit(i)
            [t_t_0, t_t_1, t_p, t_c_0, t_c_1, j_t, j_c] = ret
            # 计算tao
            self.param.tao[i] = self.param.t_c_1[i] - self.param.t[i]  # 论文假定返回数据很小，所以传输时间忽略

            # # 正式代码里，注释掉这里，把下面if中的注释取消
            # print(i)
            # cost = self.param.getAdditionCost(i=i, L=ret)
            # print(f'b(tao){self.param.b[i](self.param.tao[i])}, cost: {cost}')

            if self.param.p_w[j_t][t_t_0] == self.param.L[0] or self.param.p_f[j_c][t_c_0] == self.param.L[1]:
            # if False:
                # 用成本来判断是否应该接收
                cost = self.param.getAdditionCost(i=i, L=ret)
                # 通过用户效用判断是否接收
                [self.param.u[i], self.param.p[i]] = self.param.getUtility(i)
                # 接受任务的判决条件：用户效用为正, 且没超出资源上限
                isAccepted = self.param.u[i] > 0 and self.param.b[i](self.param.tao[i]) > cost
            else:
                # 通过用户效用判断是否接收
                [self.param.u[i], self.param.p[i]] = self.param.getUtility(i)
                # 接受任务的判决条件：用户效用为正, 且没超出资源上限
                isAccepted = self.param.u[i] > 0

            if isAccepted:
                isUnderCapacity = self.param.updateZk(i)  # 接受任务i后，更新各基站的资源使用情况
                if not isUnderCapacity:     # 若超出上限，则不会完成Zk的更新，并且不接收该任务
                    self.param.x[i] = 0
                    logs_socialwelfare[i] = self.param.getSocialWelfare()
                    continue
                self.param.x[i] = 1  # 任务i被接收
                # print(f'Task {i} is accepted!')
                for k in range(3):
                    self.updatePrice(k=k, j=j_t)
                    self.updatePrice(k=k, j=j_c)
            else:
                self.param.x[i] = 0  # 任务i被拒绝，不需要对x_c、x_t进行初始化，因为不会再用到这俩参数

            logs_socialwelfare[i] = self.param.getSocialWelfare()

        print("Online Algrithm:")
        # print("unit price of bandwidth", self.param.p_w)
        # print("unit price of cpu", self.param.p_f)
        # print("unit price of storage", self.param.p_s)
        print(f"User Satisfaction is {np.sum(self.param.x)}")
        # print("x", self.param.x)
        # print("user utility", self.param.u)
        # print("price", self.param.p)
        print("social welfare", self.param.getSocialWelfare())

        self.final_social_welfare = self.param.getSocialWelfare()
        self.final_user_satisfaction = np.sum(self.param.x)/self.param.N

        end = time.time()
        self.execute_time = end - start

        return logs_socialwelfare

    '''计算资源调度'''
    def schedulingAlgorithm_jit(self, i: int):
        self.param.x[i] = 0
        for j in range(self.param.M):
            self.param.x_t[i][j] = 0
            self.param.x_c[i][j] = 0

        M = self.param.M
        T = self.param.T
        ti = self.param.t[i]
        tmi = self.param.tao_max[i]
        
        si = self.param.s[i]
        wi = self.param.w[i]
        delta_t = self.param.delta_t
        Pi = self.param.P[i]
        h_U_i = self.param.h_U[i]
        sigma_2 = self.param.sigma_2
        p_w = self.param.p_w
        p_s = self.param.p_s
        p_f = self.param.p_f
        
        C0 = self.param.C[0]
        C1 = self.param.C[1]
        C2 = self.param.C[2]
        z0 = self.param.z[0]
        z1 = self.param.z[1]
        z2 = self.param.z[2]
        
        self.param.f[i] = wi / delta_t
        fi = self.param.f[i]
        
        phi, t_p, t_t_i, c_t_i, t_c_i, c_c_i, j_t_i, j_c_i = cal_schedule(M, T, ti, tmi, si, wi, delta_t, Pi, h_U_i, sigma_2, p_w, p_s, p_f, C0, C1, C2, z0, z1, z2, fi)
        
        self.param.fi[i] = phi
        self.param.t_p[i] = t_p

        t_p_i = self.param.t_p[i]
        j_t = j_t_i[t_p_i]
        j_c = j_c_i[t_p_i]
        self.param.x_t[i][j_t] = 1
        self.param.x_c[i][j_c] = 1
        self.param.t_t_0[i] = t_t_i[j_t][t_p_i]
        self.param.t_t_1[i] = t_t_i[j_t][t_p_i] + 1
        self.param.t_c_0[i] = t_c_i[j_c][t_p_i]
        self.param.t_c_1[i] = t_c_i[j_c][t_p_i] + 1
        return self.param.t_t_0[i], self.param.t_t_1[i], self.param.t_p[i], self.param.t_c_0[i], self.param.t_c_1[i], j_t, j_c

    def schedulingAlgorithm(self, i: int):
        self.param.x[i] = 0
        for j in range(self.param.M):
            self.param.x_t[i][j] = 0
            self.param.x_c[i][j] = 0
        t_t_i = np.array([[0 for _ in range(self.param.T)] for _ in range(self.param.M)])
        c_t_i = np.array([[1e19 for _ in range(self.param.T)] for _ in range(self.param.M)])
        t_c_i = np.array([[0 for _ in range(self.param.T)] for _ in range(self.param.M)])
        c_c_i = np.array([[1e19 for _ in range(self.param.T)] for _ in range(self.param.M)])
        j_t_i = np.array([0 for _ in range(self.param.T)])
        j_c_i = np.array([0 for _ in range(self.param.T)])

        # 这里的上下限做过修改，与论文中有点不同
        # 枚举一个中间时刻作为t_p，用于分开计算传输和计算过程的开销
        for t_p in range(self.param.t[i], min(self.param.T, self.param.t[i] + self.param.tao_max[i])):
            # 计算传输到计算站后，存在缓存中直到中间时刻t的开销
            for j in range(self.param.M):
                self.param.fi[i] = self.param.s[i] / self.param.delta_t / np.log2(1. + self.param.P[i] * self.param.h_U[i][j] / self.param.sigma_2[j])
                # 先枚举开始向通信站传输的时间点t_t_0
                for t_t_0 in range(self.param.t[i], t_p):    # t_p >= t_t_1 (t_t_1 = t_t_0 + 1)
                    # 避免超出基站j在t时刻资源k的上限
                    if self.param.C[0][j] < self.param.z[0][j][t_t_0] + self.param.fi[i]:
                        continue
                    isUnderCapacity = True

                    temp = self.param.fi[i] * self.param.p_w[j][t_t_0]
                    # 再从t_t_0开始，计算存储在通信站的开销
                    for ti in range(t_t_0, t_p):
                        if self.param.C[2][j] < self.param.z[2][j][ti] + self.param.s[i]:
                            isUnderCapacity = False
                            break
                        temp += self.param.s[i] * self.param.p_s[j][ti]

                    if isUnderCapacity and temp < c_t_i[j][t_p]:
                        t_t_i[j][t_p] = t_t_0
                        c_t_i[j][t_p] = temp

            self.param.f[i] = self.param.w[i] / self.param.delta_t
            # 计算从中间时刻t_p开始，存在缓存中，直到完成计算的开销
            for j in range(self.param.M):
                # 枚举在计算站j里开始计算的时间点t_c_0
                for t_c_0 in range(t_p, min(self.param.T, self.param.t[i] + self.param.tao_max[i])):    # t_c_0 >= t_p
                    # 避免超出基站j在t时刻资源k的上限
                    if self.param.C[1][j] < self.param.z[1][j][t_c_0] + self.param.f[i]:
                        continue
                    isUnderCapacity = True

                    temp = self.param.f[i] * self.param.p_f[j][t_c_0]
                    # 再计算从中间时刻t_p到开始计算时t_c_0的缓存开销
                    for ti in range(t_p, t_c_0 + 1):
                        if self.param.C[2][j] < self.param.z[2][j][ti] + self.param.s[i]:
                            isUnderCapacity = False
                            break
                        temp += self.param.s[i] * self.param.p_s[j][ti]

                    if isUnderCapacity and temp < c_c_i[j][t_p]:
                        t_c_i[j][t_p] = t_c_0
                        c_c_i[j][t_p] = temp
            c_t_i_t = c_t_i[:, t_p]
            c_c_i_t = c_c_i[:, t_p]
            # print(c_t_i_t, c_t_c_t)
            j_t_i[t_p] = np.argmin(c_t_i_t, axis=0)     # 保存以t_p作为站间传输时间的通信站
            j_c_i[t_p] = np.argmin(c_c_i_t, axis=0)     # 保存以t_p作为站间传输时间的计算站
            # 更新真正的fi (之前的fi为计算过程中的中间值)
            self.param.fi[i] = self.param.s[i] / self.param.delta_t / np.log2(1. + self.param.P[i] * self.param.h_U[i][j_t_i[t_p]] / self.param.sigma_2[j_t_i[t_p]])

        temp = 1e19
        # print("j_t_i", j_t_i)
        # print("j_c_i", j_c_i)
        for t_p in range(self.param.t[i], min(self.param.T, self.param.t[i] + self.param.tao_max[i] + 1)):
            c_t_i_t = c_t_i[:, t_p]
            c_c_i_t = c_c_i[:, t_p]
            # if temp > c_t_i[j_t_i[t_p]][t_p] + c_c_i[j_c_i[t_p]][t_p]:
            if temp > c_t_i_t[j_t_i[t_p]] + c_c_i_t[j_c_i[t_p]]:
                temp = c_t_i_t[j_t_i[t_p]] + c_c_i_t[j_c_i[t_p]]
                self.param.t_p[i] = t_p

        t_p_i = self.param.t_p[i]
        j_t = j_t_i[t_p_i]
        j_c = j_c_i[t_p_i]
        self.param.x_t[i][j_t] = 1
        self.param.x_c[i][j_c] = 1
        self.param.t_t_0[i] = t_t_i[j_t][t_p_i]
        self.param.t_t_1[i] = t_t_i[j_t][t_p_i] + 1
        self.param.t_c_0[i] = t_c_i[j_c][t_p_i]
        self.param.t_c_1[i] = t_c_i[j_c][t_p_i] + 1
        return self.param.t_t_0[i], self.param.t_t_1[i], self.param.t_p[i], self.param.t_c_0[i], self.param.t_c_1[i], j_t, j_c

    '''更新资源k的单位价格p_k
    :param k: 0-w, 1-f, 2-s
    :param j: 基站j
    '''
    def updatePrice(self, k: int, j: int):
        if k == 0:
            L_w = self.param.L_w
            U_w = self.param.U_w
            W = self.param.C[0][j]  # C[0][j]就是 论文中的 W[j]
            for t in range(self.param.T):
                z = self.param.z[k][j][t]
                self.param.p_w[j][t] = L_w * (U_w / L_w) ** (z / W)
        if k == 1 or k == 2:
            gamma = self.param.gamma[k][j]
            seta = np.max([2., (1. + gamma) ** (1. / gamma)])
            Capacity = self.param.C[k][j]
            U = self.param.U[k]
            beta = self.param.beta[k][j]
            rou = np.max([seta * gamma / Capacity,
                          seta / (Capacity * (seta - 1.)) * np.log(U / (beta * (1. + gamma) * (Capacity ** gamma)))])
            for t in range(self.param.T):
                z = self.param.z[k][j][t]
                if z <= Capacity / seta:
                    cost = self.param.getCostGradientOfZ(z=seta * z, k=k, j=j)
                    price = cost
                else:
                    cost = self.param.getCostGradientOfZ(z=Capacity, k=k, j=j)
                    price = cost * (np.e ** (rou * (z - Capacity / seta)))
                if k == 1:
                    self.param.p_f[j][t] = price / 1e9
                else:
                    self.param.p_s[j][t] = price / 1e6

    def setSeed(self, seed):
        np.random.seed(seed)

    def set_UL(self, UL):
        # 输入U/L的比例，设置所有U
        self.param.U[0] = self.param.L[0] * UL
        self.param.U[1] = self.param.L[1] * UL
        self.param.U[2] = self.param.L[2] * UL
        self.param.U_w = self.param.L_w * UL
