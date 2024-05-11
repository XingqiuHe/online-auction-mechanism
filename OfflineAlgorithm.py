import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linprog
from Parameter import Parameter


class OfflineAlgorithm(object):
    def __init__(self, param: Parameter):
        super().__init__()
        self.param = param

    def text_create(self, file_name, msg, x=-1, y=-1, z=-1):
        folder_path = 'matlab/'
        full_path = folder_path + file_name + '.txt'
        file = open(full_path, 'w')
        if x == -1:
            file.write(str(msg) + ' ')
        else:
            for i in range(x):
                if y == -1:
                    file.write(str(msg[i])+' ')
                    continue
                for j in range(y):
                    if z == -1:
                        file.write(str(msg[i][j])+' ')
                        continue
                    for k in range(z):
                        file.write(str(msg[i][j][k])+' ')
                    file.write('\n')
                file.write('\n')
            file.close()

    '''生成参数文件，用于yalmip使用'''
    def generate_params_for_yalmip(self):
        M = self.param.M
        N = self.param.N
        T = self.param.T
        self.text_create('M', M)
        self.text_create('N', N)
        self.text_create('T', T)
        self.text_create('delta_t', self.param.delta_t)
        self.text_create('t_', self.param.t, N)
        self.text_create('P', self.param.P, N)
        self.text_create('w', self.param.w, N)
        self.text_create('s', self.param.s, N)
        self.text_create('h_u', self.param.h_U, N, M)
        self.text_create('sigma_2', self.param.sigma_2, M)
        self.text_create('beta', self.param.beta, 3, M)
        self.text_create('gamma', self.param.gamma, 3, M)
        self.text_create('C', self.param.C, 3, M)
        self.text_create('z', self.param.z, 3, M, T)
        b_arg = [self.param.b[i](0) for i in range(N)]
        self.text_create('b_arg', b_arg, N)

    '''执行离线算法'''
    def execute(self):
        """
        需要将最优化问题的变量提出来：
        x_t[i][j]:  x[i*M + j]
        x_c[i][j]:  x[N*M + i*M + j]
        t_t_0[i]:   x[2*N*M + i]
        t_p[i]:     x[2*N*M + N + i]
        t_c_0[i]:   x[2*N*M + 2*N + i]
        """
        N = self.param.N
        M = self.param.M
        T = self.param.T
        delta_t = self.param.delta_t

        def x_t(i, j, x):
            return x[i*M + j]

        def x_c(i, j, x):
            return x[N*M + i*M + j]

        # 通信站
        def j_t(i, x):
            # 遍历x_t[i]，找出最大的x_t[i][j]
            temp = -1
            jt = -1
            for j in range(M):
                if x_t(i, j, x) > temp:
                    temp = x_t(i, j, x)
                    jt = j
            if t_c_1(i, x) > T-1:   # 超出时间调度，直接抛弃
                jt = -1
            return jt

        # 计算站
        def j_c(i, x):
            # 遍历x_c[i]，找出最大的x_c[i][j]
            temp = -1
            jc = -1
            for j in range(M):
                if x_c(i, j, x) > temp:
                    temp = x_c(i, j, x)
                    jc = j
            if t_c_1(i, x) > T-1:   # 超出时间调度，直接抛弃
                jc = -1
            return jc

        # 获取任务i的发生时间
        def t(i):
            return int(round(self.param.t[i]))

        # 获取用户i的发射功率
        def P(i):
            return self.param.P[i]

        # 获取任务i需求的CPU周期数
        def w(i):
            return self.param.w[i]

        # 获取任务i的数据量
        def s(i):
            return self.param.s[i]

        # 获取通信站j_t分配给任务i的带宽
        def fi(i, x):
            j = j_t(i, x)
            if j == -1:
                return 0
            return s(i) / delta_t / np.log2(1. + P(i) * self.param.h_U[i][j] / self.param.sigma_2[j])

        # 获取计算站分配给任务i的cpu频率
        def f(i):
            return w(i) / delta_t

        # 计算一个slot内的数据量
        def r(i, x):
            j = j_t(i, x)
            if j == -1:
                return 0
            return fi(i, x) * delta_t * np.log2(1. + P(i) * self.param.h_U[i][j] / self.param.sigma_2[j])

        # 时间调度
        def t_t_0(i, x):
            return int(round(x[2 * N * M + i]))

        def t_t_1(i, x):
            # return int(round(t_t_0(i, x) + 1))
            rr = r(i, x)
            if rr == 0.:
                return int(round(t_t_0(i, x) + 1))
            return int(round(t_t_0(i, x) + np.ceil(s(i) / rr)))

        def t_p(i, x):
            return int(round(x[2 * N * M + N + i]))

        def t_c_0(i, x):
            return int(round(x[2 * N * M + 2 * N + i]))

        def t_c_1(i, x):
            # return int(round(t_c_0(i, x) + 1))
            ff = f(i)
            if ff == 0:
                return int(round(t_c_0(i, x) + 1))
            return int(round(t_c_0(i, x) + np.ceil(w(i) / (ff * delta_t))))

        # 计算任务i的延迟
        def tao(i, x):
            return int(round(t_c_1(i, x) - t(i)))

        # 名义价值方程b
        def b(i, Tao):
            return self.param.b[i](Tao)

        # 指示变量，返回用x_t和x_c计算的x
        def x1(i, x):
            ans1 = 0
            for j in range(M):
                ans1 += x_t(i, j, x)
            # for index in range(i * M, (i + 1) * M):
            #     ans1 += x[index]
            return ans1

        def x2(i, x):
            ans2 = 0
            for j in range(M):
                ans2 += x_c(i, j, x)
            # for index in range(N*M + i*M, N*M + (i+1)*M):
            #     ans2 += x[index]
            return ans2

        self.isUpdate = False  # 指示本轮迭代是否更新过资源占用情况z

        # 资源使用情况
        def z(k, j, tt, x):
            if not self.isUpdate:     # 一轮迭代中没有计算过k才会执行下面操作
                self.param.z = np.array([[[0. for _ in range(T)] for _ in range(self.param.M)] for _ in range(3)])
                for i in range(N):
                    jt = j_t(i, x)
                    jc = j_c(i, x)
                    if jc == -1 or jt == -1:  # 说明没有分配
                        continue
                    # T_t, k=0:w
                    for tt in range(t_t_0(i, x), t_t_1(i, x)):
                        self.param.z[0][jt][tt] += fi(i, x) * x_t(i, jt, x)
                    # T_c, k=1:f
                    for tt in range(t_c_0(i, x), t_c_1(i, x)):
                        self.param.z[1][jc][tt] += f(i) * x_c(i, jc, x)
                    # T_tp, k=2:s
                    for tt in range(t_t_1(i, x), t_p(i, x)):
                        self.param.z[2][jt][tt] += s(i) * x_t(i, jt, x)
                    # T_pc, k=2:s
                    for tt in range(t_p(i, x), t_c_0(i, x)):
                        self.param.z[2][jc][tt] += s(i) * x_c(i, jc, x)
                self.isUpdate = True
            return self.param.z[k][j][tt]

        # 优化目标-转换成最小化问题（取负号）
        def fun(x):
            # Social Welfare = All Tasks' Nominal Value - All APs' Operating Costs
            totalNominalValue = 0.
            for i in range(N):
                jt = j_t(i, x)
                jc = j_c(i, x)
                if jc == -1 or jt == -1:  # 说明没有分配
                    continue
                totalNominalValue += (x_t(i, jt, x) + x_c(i, jc, x)) * b(i, tao(i, x))
            totalOperatingCosts = 0.
            for k in range(3):
                for j in range(M):
                    beta = self.param.beta[k][j]
                    gamma = self.param.gamma[k][j]
                    for tt in range(T):
                        zz = z(k, j, tt, x)
                        if k == 0:
                            zz = zz/1e6
                        else:
                            zz = zz/1e9
                        totalOperatingCosts += beta * zz ** (1. + gamma)
                        # print(k, j, tt)
            self.isUpdate = False     # 避免重复计算z
            print(totalNominalValue - totalOperatingCosts)
            return -(totalNominalValue - totalOperatingCosts)

        # 优化约束
        delta = 1e-9   # 一个极小值
        cons = []
        cons += [{'type': 'ineq', 'fun': lambda x: x1(i, x)} for i in range(N)]
        cons += [{'type': 'ineq', 'fun': lambda x: 1. - x1(i, x)} for i in range(N)]
        cons += [{'type': 'eq', 'fun': lambda x: x1(i, x) - x2(i, x)} for i in range(N)]
        cons += [{'type': 'ineq', 'fun': lambda x: 1 - x1(i, x)} for i in range(N)]
        cons += [{'type': 'ineq', 'fun': lambda x: 1 - x2(i, x)} for i in range(N)]
        # cons 2
        cons += [{'type': 'ineq', 'fun': lambda x: t_p(i, x) - t_t_1(i, x)} for i in range(N)]
        # cons 4
        cons += [{'type': 'ineq', 'fun': lambda x: t_c_0(i, x) - t_p(i, x)} for i in range(N)]
        # cons 6c 6d 6e
        for kk in range(3):
            for jj in range(M):
                for ttt in range(T):
                    cons += [{'type': 'ineq', 'fun': lambda x: self.param.C[kk][jj] - z(kk, jj, ttt, x)}]

        # # cons 6f
        # cons += [{'type': 'eq', 'fun': lambda x: x1(i, x) * (x1(i, x) - 1)} for i in range(N)]
        # for ii in range(N):
        #     for jj in range(M):
        #         cons += [{'type': 'eq', 'fun': lambda x: x_t(ii, jj, x) * (x_t(ii, jj, x) - 1)}]
        # for ii in range(N):
        #     for jj in range(M):
        #         cons += [{'type': 'eq', 'fun': lambda x: x_c(ii, jj, x) * (x_c(ii, jj, x) - 1)}]
        # # cons 6b
        # cons += [{'type': 'eq', 'fun': lambda x: x1(i, x) - x2(i, x)} for i in range(N)]
        # cons 6g
        # for ii in range(N):
        #     cons += [{'type': 'ineq', 'fun': lambda x: t_t_0(ii, x) - t(ii)}]
        #     cons += [{'type': 'ineq', 'fun': lambda x: T - 1 - t_t_0(ii, x)}]
        #     cons += [{'type': 'ineq', 'fun': lambda x: t_t_1(ii, x) - t(ii)}]
        #     cons += [{'type': 'ineq', 'fun': lambda x: T - 1 - t_t_1(ii, x)}]
        #     cons += [{'type': 'ineq', 'fun': lambda x: t_p(ii, x) - t(ii)}]
        #     cons += [{'type': 'ineq', 'fun': lambda x: T - 1 - t_p(ii, x)}]
        #     cons += [{'type': 'ineq', 'fun': lambda x: t_c_0(ii, x) - t(ii)}]
        #     cons += [{'type': 'ineq', 'fun': lambda x: T - 1 - t_c_0(ii, x)}]
        #     cons += [{'type': 'ineq', 'fun': lambda x: t_c_1(ii, x) - t(ii)}]
        #     cons += [{'type': 'ineq', 'fun': lambda x: T - 1 - t_c_1(ii, x)}]

        bounds = [(0, 1) for _ in range(2*N*M + 3*N)]
        # for index in range(2*N*M):
        #     bounds[index] = [(0 - delta, 1 + delta)]
        for ii in range(N):
            index1 = ii + 2*N*M
            bounds[index1] = (t(ii), max(t(ii), T-3))
            index2 = ii + 2*N*M + N
            bounds[index2] = (t(ii), max(t(ii)+1, T-2))
            index3 = ii + 2*N*M + 2*N
            bounds[index3] = (t(ii), max(t(ii)+1, T-2))

        # 初始情况，需要满足约束（修改t相关项）
        x0 = [0. for _ in range(2*N*M + 3*N)]
        for ii in range(N):
            """
            x_t[i][j]:  x[i*M + j]
            x_c[i][j]:  x[N*M + i*M + j]
            """
            if not np.random.randint(0, M+1):   # 以1/(M+1)的几率不选择该任务
                continue
            isFesible = False
            counter = 0
            while not isFesible:
                counter += 1
                if counter >= 2*M:  # 太多次尝试失败则不选择该任务
                    break
                jj_t = np.random.randint(0, M)
                x0[ii * M + jj_t] = 1.
                self.isUpdate = False
                isFesible = True
                for kk in range(3):
                    for jj in range(M):
                        for ttt in range(T):
                            if self.param.C[kk][jj] < z(kk, jj, ttt, x0):
                                isFesible = False
                            if not isFesible:
                                break
                        if not isFesible:
                            break
                    if not isFesible:
                        break
                if not isFesible:
                    x0[ii * M + jj_t] = 0.
            if counter >= 2 * M:  # 太多次尝试失败则不选择该任务
                continue
            isFesible = False
            counter = 0
            while not isFesible:
                counter += 1
                if counter >= 2 * M:  # 太多次尝试失败则不选择该任务
                    break
                jj_c = np.random.randint(0, M)
                x0[N*M + ii*M + jj_c] = 1.
                self.isUpdate = False
                isFesible = True
                for kk in range(3):
                    for jj in range(M):
                        for ttt in range(T):
                            if self.param.C[kk][jj] < z(kk, jj, ttt, x0):
                                isFesible = False
                            if not isFesible:
                                break
                        if not isFesible:
                            break
                    if not isFesible:
                        break
                if not isFesible:
                    x0[N*M + ii*M + jj_c] = 0.

        for ii in range(N):
            """
            t_t_0[i]:   x[2*N*M + i]
            t_p[i]:     x[2*N*M + N + i]
            t_c_0[i]:   x[2*N*M + 2*N + i]
            """
            x0[2 * N * M + ii] = t(ii)
            x0[2 * N * M + N + ii] = t_t_1(ii, x0)
            x0[2 * N * M + 2 * N + ii] = t_p(ii, x0)

        # print(fun)
        # print(x0)
        # print(cons)
        # print(bounds)

        for index in range(2*N*M + 3*N):
            if x0[index] < bounds[index][0] or x0[index] > bounds[index][1]:
                print("BOUNDS ERROR:", index, x0[index], bounds[index][0], bounds[index][1])

        '''
        问题一：约束存在冲突问题
            从常理上讲，若t[i] >= T-2（计算完成时间必然超过总时间T），则必定不能选择任务i。但若用scipy minimize的constraint来限制t_c_1的上限，则会导致相关参数的上下限约束冲突，同样bounds也会导致该问题
            目前的解决方案：放松bounds的上限，并在获取jc、jt处直接把这类越界的任务抛弃
        问题二：所有不存在"问题一"的任务都会被"接纳"
            由松弛指示变量约束造成的，很少有任务会被抛弃，但也可以看成以多高的概率采用某个调度
            目前的解决方案：修改了目标函数，让用户预期变成 (选择jc的概率+选择jt的概率)*用户名义价值
        问题三：收敛效果不好
            经过调试，发现在资源上限处存在问题，如果对基站的资源上限进行了约束，我们松弛指示变量的条件就会导致始终超出资源上限。并且因为minimize的特性，如果不从一开始手动选择一个比较好的初始参数，它根本跨不出一步，在我们这里就表现成一直呆在全0的点跑不出一步
            因此，我认为简单的放松约束并没有多好的效果，恐怕枚举才是最优解
        问题四：我发现分配给任务i的带宽fi总是算出来贼大，不是很懂为什么fi计算公式中，增益越大带宽也越大
        '''

        res = minimize(fun=fun, x0=np.array(x0), constraints=cons, bounds=bounds)

        print(res)