import numpy as np
import copy
from Parameter import Parameter
import time

class BenchmarkAlgorithm(object):
    def __init__(self, param: Parameter):
        super().__init__()
        self.randomParam = copy.deepcopy(param)
        self.greedyParam = copy.deepcopy(param)
        self.FIFOParam = copy.deepcopy(param)
        self.final_social_welfare = [0 for _ in range(3)]
        self.final_user_satisfaction = [0 for _ in range(3)]
        self.execute_time = [0 for _ in range(3)]

    def runBenchmark(self):
        params = [self.randomParam, self.greedyParam, self.FIFOParam]
        logs = []
        for algID in range(3):
            start = time.time()
            logs.append(self.execute(param=params[algID], algID=algID))
            end = time.time()
            self.execute_time[algID] = end - start

            self.final_social_welfare[algID] = params[algID].getSocialWelfare()
            self.final_user_satisfaction[algID] = np.sum(params[algID].x)/params[algID].N
        return logs

    '''Random online mechanism: 
    For each requester, the BS randomly selects the task executor 
    and randomly schedules the transmission and computation times.
    '''
    def randomScheduling(self, i: int, param: Parameter):
        param.x[i] = 0
        for j in range(param.M):
            param.x_t[i][j] = 0
            param.x_c[i][j] = 0

        random_times = 5
        ans_jt = -1
        ans_jc = -1
        ans_tt0 = -1
        ans_tc0 = -1
        # 随机选择一个时间点t_p
        for counter_tp in range(random_times):
            if min(param.tao_max[i], param.T - param.t[i]) < 2:
                break
            t_p = np.random.randint(param.t[i] + 1, min(param.T, param.t[i] + param.tao_max[i]))

            isUnderCapacity = True
            # 随机选择一个通信站j_t
            for counter_jt in range(random_times):
                j_t = np.random.randint(0, param.M)
                fi = param.s[i] / param.delta_t / np.log2(1. + param.P[i] * param.h_U[i][j_t] / param.sigma_2[j_t])
                # 随机选择一个开始向通信站传输的时间点t_t_0
                for counter_tt0 in range(random_times):
                    t_t_0 = np.random.randint(param.t[i], t_p)  # t_p >= t_t_1 (t_t_1 = t_t_0 + 1)
                    # 避免超出基站j在t时刻资源k的上限
                    if param.C[0][j_t] < param.z[0][j_t][t_t_0] + fi:
                        continue
                    isUnderCapacity = True
                    # 避免超出基站j在ti时刻的带宽上限
                    for ti in range(t_t_0, t_p):
                        if param.C[2][j_t] < param.z[2][j_t][ti] + param.s[i]:
                            isUnderCapacity = False
                            break
                    if isUnderCapacity:
                        ans_tt0 = t_t_0
                        break

                if isUnderCapacity:
                    ans_jt = j_t
                    break

            if not isUnderCapacity:
                continue

            param.f[i] = param.w[i] / param.delta_t
            # 随机选择一个通信站j_c
            for counter_jc in range(random_times):
                j_c = np.random.randint(0, param.M)

                # 枚举在计算站j里开始计算的时间点t_c_0
                for counter_tc in range(random_times):
                    t_c_0 = np.random.randint(t_p, min(param.T, param.t[i] + param.tao_max[i]))  # t_c_0 >= t_p
                    # 避免超出基站j在t时刻资源k的上限
                    if param.C[1][j_c] < param.z[1][j_c][t_c_0] + param.f[i]:
                        continue
                    isUnderCapacity = True
                    # 避免超出基站j在ti时刻的带宽上限
                    for ti in range(t_p, t_c_0 + 1):
                        if param.C[2][j_c] < param.z[2][j_c][ti] + param.s[i]:
                            isUnderCapacity = False
                            break
                    if isUnderCapacity:
                        ans_tc0 = t_c_0
                        break

                if isUnderCapacity:
                    ans_jc = j_c
                    break

            if isUnderCapacity:
                j_t = ans_jt
                j_c = ans_jc
                # 更新真正的fi (之前的fi为计算过程中的中间值)
                param.fi[i] = param.s[i] / param.delta_t / np.log2(
                    1. + param.P[i] * param.h_U[i][j_t] / param.sigma_2[j_t])
                # 更新参数
                param.x_t[i][j_t] = 1
                param.x_c[i][j_c] = 1
                param.t_t_0[i] = ans_tt0
                param.t_t_1[i] = ans_tt0 + 1
                param.t_p[i] = t_p
                param.t_c_0[i] = ans_tc0
                param.t_c_1[i] = ans_tc0 + 1
                break
        return param.t_t_0[i], param.t_t_1[i], param.t_p[i], param.t_c_0[i], param.t_c_1[i], ans_jt, ans_jc

    '''Greedy online mechanism: 
    Upon the arrival of a requester, the BS chooses the task executor
    with the maximal valuation as the winner and schedules one time
    slot for transmission and ⌈ τ ⌉ − 1 time slots for computation.
    '''
    def greedyScheduling(self, i: int, param: Parameter):
        param.x[i] = 0
        for j in range(param.M):
            param.x_t[i][j] = 0
            param.x_c[i][j] = 0

        # 初始化一个随机的价格

        t_t_i = np.array([[0 for _ in range(param.T)] for _ in range(param.M)])
        c_t_i = np.array([[1e19 for _ in range(param.T)] for _ in range(param.M)])
        t_c_i = np.array([[0 for _ in range(param.T)] for _ in range(param.M)])
        c_c_i = np.array([[1e19 for _ in range(param.T)] for _ in range(param.M)])
        j_t_i = np.array([0 for _ in range(param.T)])
        j_c_i = np.array([0 for _ in range(param.T)])

        # 这里的上下限做过修改，与论文中有点不同
        # 枚举一个中间时刻作为t_p，用于分开计算传输和计算过程的开销
        for t_p in range(param.t[i], min(param.T, param.t[i] + param.tao_max[i])):
            # 计算传输到计算站后，存在缓存中直到中间时刻t的开销
            for j in range(param.M):
                param.fi[i] = param.s[i] / param.delta_t / np.log2(
                    1. + param.P[i] * param.h_U[i][j] / param.sigma_2[j])
                # 先枚举开始向通信站传输的时间点t_t_0
                for t_t_0 in range(param.t[i], t_p):  # t_p >= t_t_1 (t_t_1 = t_t_0 + 1)
                    # 避免超出基站j在t时刻资源k的上限
                    if param.C[0][j] < param.z[0][j][t_t_0] + param.fi[i]:
                        continue
                    isUnderCapacity = True

                    temp = param.fi[i] * param.p_w[j][t_t_0]
                    # 再从t_t_0开始，计算存储在通信站的开销
                    for ti in range(t_t_0, t_p):
                        if param.C[2][j] < param.z[2][j][ti] + param.s[i]:
                            isUnderCapacity = False
                            break
                        temp += param.s[i] * param.p_s[j][ti]

                    if isUnderCapacity and temp < c_t_i[j][t_p]:
                        t_t_i[j][t_p] = t_t_0
                        c_t_i[j][t_p] = temp

            param.f[i] = param.w[i] / param.delta_t
            # 计算从中间时刻t_p开始，存在缓存中，直到完成计算的开销
            for j in range(param.M):
                # 枚举在计算站j里开始计算的时间点t_c_0
                for t_c_0 in range(t_p, min(param.T, param.t[i] + param.tao_max[i])):  # t_c_0 >= t_p
                    # 避免超出基站j在t时刻资源k的上限
                    if param.C[1][j] < param.z[1][j][t_c_0] + param.f[i]:
                        continue
                    isUnderCapacity = True

                    temp = param.f[i] * param.p_f[j][t_c_0]
                    # 再计算从中间时刻t_p到开始计算时t_c_0的缓存开销
                    for ti in range(t_p, t_c_0 + 1):
                        if param.C[2][j] < param.z[2][j][ti] + param.s[i]:
                            isUnderCapacity = False
                            break
                        temp += param.s[i] * param.p_s[j][ti]

                    if isUnderCapacity and temp < c_c_i[j][t_p]:
                        t_c_i[j][t_p] = t_c_0
                        c_c_i[j][t_p] = temp
            c_t_i_t = c_t_i[:, t_p]
            c_c_i_t = c_c_i[:, t_p]
            # print(c_t_i_t, c_t_c_t)
            j_t_i[t_p] = np.argmin(c_t_i_t, axis=0)  # 保存以t_p作为站间传输时间的通信站
            j_c_i[t_p] = np.argmin(c_c_i_t, axis=0)  # 保存以t_p作为站间传输时间的计算站
            # 更新真正的fi (之前的fi为计算过程中的中间值)
            param.fi[i] = param.s[i] / param.delta_t / np.log2(1. + param.P[i] * param.h_U[i][j_t_i[t_p]] / param.sigma_2[j_t_i[t_p]])

        temp = 1e19
        # print("j_t_i", j_t_i)
        # print("j_c_i", j_c_i)
        for t_p in range(param.t[i], min(param.T, param.t[i] + param.tao_max[i] + 1)):
            c_t_i_t = c_t_i[:, t_p]
            c_c_i_t = c_c_i[:, t_p]
            # if temp > c_t_i[j_t_i[t_p]][t_p] + c_c_i[j_c_i[t_p]][t_p]:
            if temp > c_t_i_t[j_t_i[t_p]] + c_c_i_t[j_c_i[t_p]]:
                temp = c_t_i_t[j_t_i[t_p]] + c_c_i_t[j_c_i[t_p]]
                param.t_p[i] = t_p

        t_p_i = param.t_p[i]
        j_t = j_t_i[t_p_i]
        j_c = j_c_i[t_p_i]
        param.x_t[i][j_t] = 1
        param.x_c[i][j_c] = 1
        param.t_t_0[i] = t_t_i[j_t][t_p_i]
        param.t_t_1[i] = t_t_i[j_t][t_p_i] + 1
        param.t_c_0[i] = t_c_i[j_c][t_p_i]
        param.t_c_1[i] = t_c_i[j_c][t_p_i] + 1
        return param.t_t_0[i], param.t_t_1[i], param.t_p[i], param.t_c_0[i], param.t_c_1[i], j_t, j_c

    '''更新资源k的单位价格p_k
    :param k: 0-w, 1-f, 2-s
    :param j: 基站j
    '''
    def updatePrice(self, k: int, j: int, param: Parameter):
        # # 用当前的使用情况来计价
        # for t in range(param.T):
        #     z = param.z[k][j][t]
        #     c = param.getCostGradientOfZ(z=z, k=k, j=j)
        #     if k == 1:
        #         param.p_f[j][t] = c
        #     else:
        #         param.p_s[j][t] = c
        # L_w = param.L_w
        # U_w = param.U_w
        # W = param.C[0][j]  # C[0][j]就是 论文中的 W[j]
        # for t in range(param.T):
        #     z = param.z[k][j][t]
        #     param.p_w[j][t] = L_w * (U_w / L_w) ** (z / W)
        if k == 0:
            L_w = param.L_w
            U_w = param.U_w
            W = param.C[0][j]  # C[0][j]就是 论文中的 W[j]
            for t in range(param.T):
                z = param.z[k][j][t]
                param.p_w[j][t] = L_w * (U_w / L_w) ** (z / W)
        if k == 1 or k == 2:
            for t in range(param.T):
                z = param.z[k][j][t]
                cost = param.getCostGradientOfZ(z=z, k=k, j=j)
                price = cost
                if k == 1:
                    param.p_f[j][t] = price / 1e9
                else:
                    param.p_s[j][t] = price / 1e6

    '''First In First Out (FIFO) online mechanism:
     Arriving tasks are always accepted with a ﬁxed transmission and
      computation time schedule till the resources are run out.
    '''
    def FIFOScheduling(self, i: int, param: Parameter):
        param.x[i] = 0
        for j in range(param.M):
            param.x_t[i][j] = 0
            param.x_c[i][j] = 0

        t_t_0 = param.t[i]  # 每一个task过来，尝试立即传输
        t_t_1 = t_t_0 + 1
        t_p = t_t_1
        t_c_0 = t_p
        t_c_1 = t_c_0 + 1
        j_t = -1
        j_c = -1

        if t_c_1 < min(param.T, param.t[i] + param.tao_max[i]):
            # 传输节点是t_t_0可用的、当前task的通信范围内信道增益最好的AP
            h_u_i = param.h_U[i]
            j_t = np.argmax(h_u_i)
            fi = param.s[i] / param.delta_t / np.log2(1. + param.P[i] * param.h_U[i][j_t] / param.sigma_2[j_t])
            while param.C[0][j_t] < param.z[0][j_t][t_t_0] + fi:
                h_u_i[j_t] = -1
                j_t = np.argmax(h_u_i)
                fi = param.s[i] / param.delta_t / np.log2(1. + param.P[i] * param.h_U[i][j_t] / param.sigma_2[j_t])
                if h_u_i[j_t] == -1:
                    j_t = -1
                    break

            # 算节点是t_c_0时剩余计算资源最多的AP
            z_remain = param.C[1] - param.z[1, :, t_c_0]
            j_c = np.argmax(z_remain)
            param.f[i] = param.w[i] / param.delta_t
            while param.C[1][j_c] < param.z[1][j_c][t_c_0] + param.f[i]:
                z_remain[j_c] = -1
                j_c = np.argmax(z_remain)
                if z_remain[j_c] == -1:
                    j_c = -1
                    break

        if j_t != -1 and j_c != -1:
            param.fi[i] = param.s[i] / param.delta_t / np.log2(1. + param.P[i] * param.h_U[i][j_t] / param.sigma_2[j_t])
            param.x_t[i][j_t] = 1
            param.x_c[i][j_c] = 1
            param.t_t_0[i] = t_t_0
            param.t_t_1[i] = t_t_1
            param.t_p[i] = t_p
            param.t_c_0[i] = t_c_0
            param.t_c_1[i] = t_c_1

        return param.t_t_0[i], param.t_t_1[i], param.t_p[i], param.t_c_0[i], param.t_c_1[i], j_t, j_c

    '''执行算法
    :parameter algID: benchmark算法的号码，1对应随机算法，2对应贪婪算法，3对应FIFO算法
    '''
    def execute(self, param: Parameter, algID: int):
        # initial parameters
        for i in range(param.N):
            param.x[i] = 0
            for j in range(param.M):
                param.x_t[i][j] = 0
                param.x_c[i][j] = 0
        for j in range(param.M):
            for t in range(param.T):
                # k=0: w
                param.p_w[j][t] = param.L_w
                # k=1: f
                param.p_f[j][t] = param.L[1]
                # k=2: s
                param.p_s[j][t] = param.L[2]

        logs_socialwelfare = [0. for _ in range(param.N)]

        # when task i comes
        for i in range(param.N):
            # 计算tao_max
            temp = 1
            while param.b[i](temp) >= 0:
                temp += 1
            param.tao_max[i] = temp - 1

            if algID == 0:
                ret = self.randomScheduling(i, param)
            elif algID == 1:
                ret = self.greedyScheduling(i, param)
            else:
                ret = self.FIFOScheduling(i, param)

            j_t = ret[5]  # i的通信站
            j_c = ret[6]  # i的计算站
            if j_t == -1 or j_c == -1:
                logs_socialwelfare[i] = param.getSocialWelfare()
                continue
            # 计算tao
            param.tao[i] = param.t_c_1[i] - param.t[i]  # 论文假定返回数据很小，所以传输时间忽略

            # if algID == 0 or algID == 2:
            #     # 用成本来判断是否应该接收
            #     cost = param.getAdditionCost(i=i, L=ret)
            #     isAccepted = param.b[i](param.tao[i]) > cost
            # else:
            #     # 通过用户效用判断是否接收
            #     [param.u[i], param.p[i]] = param.getUtility(i)
            #     # 接受任务的判决条件：用户效用为正, 且没超出资源上限
            #     isAccepted = param.u[i] > 0
            # 用成本来判断是否应该接收
            cost = param.getAdditionCost(i=i, L=ret)
            isAccepted = param.b[i](param.tao[i]) > cost

            if isAccepted:
                isUnderCapacity = param.updateZk(i)  # 接受任务i后，更新各基站的资源使用情况
                if not isUnderCapacity:  # 若超出上限，则不会完成Zk的更新，并且不接收该任务
                    param.x[i] = 0
                    logs_socialwelfare[i] = param.getSocialWelfare()
                    continue
                param.x[i] = 1  # 任务i被接收
                # if algID == 1:
                for k in range(3):
                    self.updatePrice(k=k, j=j_t, param=param)
                    self.updatePrice(k=k, j=j_c, param=param)
            else:
                param.x[i] = 0

            logs_socialwelfare[i] = param.getSocialWelfare()

        if algID == 0:
            print(f"Benchmark {algID} - RandomAlg:")
        elif algID == 1:
            print(f"Benchmark {algID} - GreedyAlg:")
        else:
            print(f"Benchmark {algID} - FIFOAlg:")
        print(f"User Satisfaction is {np.sum(param.x)}")
        # print("x", param.x)
        # print("user utility", param.u)
        # print("price", param.p)
        print("social welfare", param.getSocialWelfare())

        return logs_socialwelfare

    def setSeed(self, seed):
        np.random.seed(seed)
