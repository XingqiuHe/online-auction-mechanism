import numpy as np

'''Parameter List:
    M              = 15                                   ; number of APs
    N              = 1500                                 ; number of TASKs
    T              = 20                                   ; number of time slots
    delta_t        = 1 s                                  ; length of a time slot

    t[i]           = randomly in [0, T]                   ; arrival time of task
    s[i]           = randomly from 10 to 30 MB            ; task size
    w[i]           = 100 cycles/Byte * s[i]               ; required CPU cycles (calculated by CPU cycles coefficient * task size)
    b[i](x)        = [11, 90) - [1, 10) * x               ; nominal value 
    P[i]           = 1.5W                                 ; transmission power at requesters
    
    sigma_2[j]     = 10^(-9) W                            ; noise power at APs
    L_w[j]         = 1 * 1e-6                             ; lowest value of one unit of spectrum bandwidth
    U_w[j]         = 50 * 1e-6                            ; highest value of one unit of spectrum bandwidth
    U[k]           = 50 (start from 1)                    ; the highest price of single resource k (k = 2,3)

    beta[1][j]     = randomly in [0.4, 0.6]               ; beta of CPU
    beta[2][j]     = randomly in [0.005, 0.02]            ; beta of RAM
    gamma[1][j]    = randomly in [1.7, 2.2]               ; gamma of CPU
    gamma[2][j]    = randomly in [0.5, 1]                 ; gamma of RAM

    C[0][j]        = 40MHz                                ; Total bandwidth
    C[1][j]        = 5GBHz                                ; CPU capacity
    C[2][j]        = 5GB                                  ; Storage capacity

    Randomly generate the locations of WDs and APs in a 500m*500m square
    theta_U        = 6.25 * 10^(-4)                       ; uplink gain coefficient
    dist[i][k]     = random                               ; distance between WD i and AP j
    h_random[i][j] = CN(0,1)                              ; complex normal distribution（直接用虚数的模）
    h_U[i][j]      = theta_U * dist[i][k]^(-3) * h_random ; uplink channel gain
    h_D[i][j]      = 2 * h_U[i][j]                        ; downlink channel gain

基本单位设置为s(delta_t)、Byte(Storage, 单位价格需要/1e6)、Hz(CPU, 单位价格需要/1e9)、Hz(Bandwidth, 单位价格需要/1e6)、W(Power)
资源价格上限U[k]用的是paper 1(Ruiting Zhou)的设置，其他的参数基本来自paper 2(Gang Li)，可能需要调整U
名义价值方程用的是简单的线性方程，可能需要更改
信道增益的设置继承自WPMEC那篇文章

不同于论文，这里根据参考论文设置了一个单位价格的下界(1e-9，在线算法的初始化过程中修改了p_s与p_f的初值)
通过调试发现，如果tao_max过于宽松，在任务数量N没有远大于基站数量M的情况下，极其容易出现social welfare为负数的情况（这是合理的，因为对于N<M*T而言，这批用户收取的费用极低，对基站而言是亏本的）
'''


class Parameter(object):
    def __init__(self, N=1500, M=15, T=20):
        super().__init__()

        self.delta_t = 1                                    # 时隙slot
        self.T = T                                          # 一局仿真的时隙数量
        self.N = N                                          # 任务集N，任务由0到N-1进行标号，只需存储任务数量N即可
        self.M = M                                          # 基站集M，任务由0到M-1进行标号，只需存储基站数量M即可

        '''基站'''
        self.sigma_2 = np.array([10.**(-9) for _ in range(self.M)])                  # 基站j的噪声功率，sigma^2
        self.h_U = np.array([[0. for _ in range(self.M)] for _ in range(self.N)])    # 用户i与基站j间的上行增益
        self.locUE = np.array([[0., 0.] for _ in range(self.N)])                    # locUE[i] 是 UE i 在图中的横纵坐标
        self.locAP = np.array([[0., 0.] for _ in range(self.M)])                    # locAP[j] 是 AP j 在图中的横纵坐标

        self.L_w = 0.               # lowest value of one unit of spectrum bandwidth
        self.U_w = 0.               # highest value of one unit of spectrum bandwidth

        '''任务基础信息'''
        self.t = np.array([0 for _ in range(self.N)])        # 任务i的到达时间
        self.s = np.array([0. for _ in range(self.N)])       # 任务i的比特量   # 为任务i分配的存储资源
        self.w = np.array([0 for _ in range(self.N)])        # 任务i需求的CPU周期数
        self.b = [lambda x: 10. - 1. * x for _ in range(self.N)]  # 任务i的名义价值方程（用户对任务的预期估价）

        self.tao = np.array([0 for _ in range(self.N)])      # 任务i的响应时间
        self.tao_max = np.array([0 for _ in range(self.N)])  # 任务i的最大响应时间（名义价值大于最小开销的响应时间长度）

        '''指示变量'''
        self.x_t = np.array([[0 for _ in range(self.M)] for _ in range(self.N)])    # 指示任务i是否向基站j通信
        self.x_c = np.array([[0 for _ in range(self.M)] for _ in range(self.N)])    # 指示任务i是否被基站j处理
        self.x = np.array([0 for _ in range(self.N)])                               # 指示任务i是否被接受

        '''任务属性'''
        self.fi = np.array([0. for _ in range(self.N)])      # 为任务i分配的带宽
        self.f = np.array([0. for _ in range(self.N)])       # 为任务i分配的CPU频率

        self.P = np.array([0. for _ in range(self.N)])       # 任务i的发送功率
        self.r = np.array([0. for _ in range(self.N)])       # 任务i在一个时隙内的数据传输量
        self.p = np.array([0. for _ in range(self.N)])       # 任务i在CEC中需缴纳的费用
        self.u = np.array([0. for _ in range(self.N)])       # 用户效用（被采纳时，用户效用=预期估价-费用，基站效用=费用-开销）

        '''任务的时间参数'''
        self.t_t_0 = np.array([0 for _ in range(self.N)])   # 任务i的传输开始时间（从用户侧发给通信站）
        self.t_t_1 = np.array([0 for _ in range(self.N)])   # 任务i的传输完毕时间
        # t_t_1[i] - t_t_0[i] = np.ceil(s[i] / r[i])
        self.t_p = np.array([0 for _ in range(self.N)])     # 通信站开始向计算站传输的时间（两站之间传输时间几乎可忽略）
        self.t_c_0 = np.array([0 for _ in range(self.N)])   # 任务i的计算开始时间（基本上等于t_p）
        self.t_c_1 = np.array([0 for _ in range(self.N)])   # 任务i的计算完毕时间
        # t_c_1[i] = t_c_0[i] + np.ceil(w[i] / (f[i] * delta_t))

        '''基站资源 K：0-传输带宽，1-CPU频率，2-存储资源'''
        self.z = np.array([[[0. for _ in range(T)] for _ in range(self.M)] for _ in range(3)])
        # z[k][j][t]: 基站j在t时隙下，资源k的使用量
        self.C = np.array([[0. for _ in range(self.M)] for _ in range(3)])
        # C[k][j]: 基站j中资源k的上限
        self.beta = np.array([[0. for _ in range(self.M)] for _ in range(3)])
        self.gamma = np.array([[0. for _ in range(self.M)] for _ in range(3)])
        self.U = np.array([0. for _ in range(3)])           # 资源k的单位价格上限
        self.L = np.array([0. for _ in range(3)])           # 资源k的单位价格下限

        '''对偶问题'''
        self.p_w = np.array([[0. for _ in range(T)] for _ in range(self.M)])         # p_w[j][t]是基站j在时隙t下单位带宽的价格
        self.p_f = np.array([[0. for _ in range(T)] for _ in range(self.M)])         # p_w[j][t]是基站j在时隙t下单位频率的价格
        self.p_s = np.array([[0. for _ in range(T)] for _ in range(self.M)])         # p_w[j][t]是基站j在时隙t下单位存储的价格

    '''计算Social Welfare'''
    def getSocialWelfare(self):
        # Social Welfare = All Tasks' Nominal Value - All APs' Operating Costs
        totalNominalValue = 0.
        for i in range(self.N):
            totalNominalValue += self.x[i] * self.b[i](self.tao[i])
        totalOperatingCosts = 0.
        for k in range(3):
            for j in range(self.M):
                for t in range(self.T):
                    totalOperatingCosts += self.getCostOfZ(z=self.z[k][j][t], k=k, j=j)
        # print(f'Nominal: {totalNominalValue}, Operating: {totalOperatingCosts}')
        return totalNominalValue - totalOperatingCosts

    '''计算基站j在t时刻下资源k的开销
    :param z: t时刻下资源k的使用量，不从self.z中取是为了保证计算预期使用量的时候能够复用该方法
    :param k: 资源种类k={0, 1, 2}
    :param j: 基站j
    '''
    def getCostOfZ(self, z: float, k: int, j: int):
        beta = self.beta[k][j]
        gamma = self.gamma[k][j]
        if self.C[k][j] >= z >= 0:
            if k == 1:
                z = z / 1e9
            else:
                z = z / 1e6
            return beta * (z**(1. + gamma))
        return 1e19      # 代表正无穷

    '''计算基站j在t时刻下资源k开销的梯度值
    :param z: t时刻下资源k的使用量，不从self.z中取是为了保证计算预期使用量的时候能够复用该方法
    :param k: 资源种类k={0, 1, 2}
    :param j: 基站j
    '''
    def getCostGradientOfZ(self, z: float, k: int, j: int):
        beta = self.beta[k][j]
        gamma = self.gamma[k][j]
        if self.C[k][j] >= z >= 0:
            if k == 1:
                z = z / 1e9
            else:
                z = z / 1e6
            return beta * (1. + gamma) * (z**gamma)
        return 1e19      # 代表正无穷

    '''获取任务i的用户效用和需要缴纳的价格'''
    def getUtility(self, i: int):
        # 这里我是取的[t1, t2)，上界应该是取不到的吧？但是论文中是[,]
        b_i_l = self.b[i](self.tao[i])
        c_i_l = 0.
        j_t = np.argmax(self.x_t[i])    # i的通信站
        j_c = np.argmax(self.x_c[i])    # i的计算站
        # T_t
        for t in range(self.t_t_0[i], self.t_t_1[i]):
            c_i_l += self.fi[i] * self.p_w[j_t][t]
        # T_c
        for t in range(self.t_c_0[i], self.t_c_1[i]):
            c_i_l += self.f[i] * self.p_f[j_c][t]
        # T_tp
        for t in range(self.t_t_1[i], self.t_p[i]):
            c_i_l += self.s[i] * self.p_s[j_t][t]
        # T_pc
        for t in range(self.t_p[i], self.t_c_0[i]):
            c_i_l += self.s[i] * self.p_s[j_c][t]
        # 返回用户效用和价格
        return b_i_l - c_i_l, c_i_l

    '''接受任务i后，用i的调度更新基站的资源使用情况'''
    def updateZk(self, i: int):
        j_t = np.argmax(self.x_t[i])    # i的通信站
        j_c = np.argmax(self.x_c[i])    # i的计算站
        #先验证
        # T_t, k=0:w
        for t in range(self.t_t_0[i], self.t_t_1[i]):
            if self.z[0][j_t][t] + self.fi[i] > self.C[0][j_t]:
                return False
        # T_c, k=1:f
        for t in range(self.t_c_0[i], self.t_c_1[i]):
            if self.z[1][j_c][t] + self.f[i] > self.C[1][j_c]:
                return False
        # T_tp, k=2:s
        for t in range(self.t_t_1[i], self.t_p[i]):
            if self.z[2][j_t][t] + self.s[i] > self.C[2][j_t]:
                return False
        # T_pc, k=2:s
        for t in range(self.t_p[i], self.t_c_0[i]):
            if self.z[2][j_c][t] + self.s[i] > self.C[2][j_c]:
                return False

        # 再更新
        # T_t, k=0:w
        for t in range(self.t_t_0[i], self.t_t_1[i]):
            self.z[0][j_t][t] += self.fi[i]
        # T_c, k=1:f
        for t in range(self.t_c_0[i], self.t_c_1[i]):
            self.z[1][j_c][t] += self.f[i]
        # T_tp, k=2:s
        for t in range(self.t_t_1[i], self.t_p[i]):
            self.z[2][j_t][t] += self.s[i]
        # T_pc, k=2:s
        for t in range(self.t_p[i], self.t_c_0[i]):
            self.z[2][j_c][t] += self.s[i]

        return True

    '''获取对任务i执行调度L后会新增的基站开销
    :parameter L: list, L = [t_t_0, t_t_1, t_p, t_c_0, t_c_1, j_t, j_c]
    '''
    def getAdditionCost(self, i: int, L: list):
        [t_t_0, t_t_1, t_p, t_c_0, t_c_1, j_t, j_c] = L
        if j_t == -1 or j_c == -1:
            return 1e19
        ans = 0
        # T_t, k=0:w
        for t in range(t_t_0, t_t_1):
            k = 0
            z = self.z[k][j_t][t]
            oldValue = self.getCostOfZ(z=z, k=k, j=j_t)
            z_new = z + self.fi[i]
            newValue = self.getCostOfZ(z=z_new, k=k, j=j_t)
            ans += newValue - oldValue
        # T_c, k=1:f
        for t in range(t_c_0, t_c_1):
            k = 1
            z = self.z[k][j_c][t]
            oldValue = self.getCostOfZ(z=z, k=k, j=j_c)
            z_new = z + self.f[i]
            newValue = self.getCostOfZ(z=z_new, k=k, j=j_c)
            ans += newValue - oldValue
        # T_tp, k=2:s
        for t in range(t_t_1, t_p):
            k = 2
            z = self.z[k][j_t][t]
            oldValue = self.getCostOfZ(z=z, k=k, j=j_t)
            z_new = z + self.s[i]
            newValue = self.getCostOfZ(z=z_new, k=k, j=j_t)
            ans += newValue - oldValue
        # T_pc, k=2:s
        for t in range(t_p, t_c_0):
            k = 2
            z = self.z[k][j_c][t]
            oldValue = self.getCostOfZ(z=z, k=k, j=j_c)
            z_new = z + self.s[i]
            newValue = self.getCostOfZ(z=z_new, k=k, j=j_c)
            ans += newValue - oldValue
        return ans

    def getProfits(self):
        totalPrice = 0.
        for i in range(self.N):
            totalPrice += self.x[i] * self.getUtility(i)[1]
        totalOperatingCosts = 0.
        for k in range(3):
            for j in range(self.M):
                for t in range(self.T):
                    totalOperatingCosts += self.getCostOfZ(z=self.z[k][j][t], k=k, j=j)
        return totalPrice - totalOperatingCosts

    def setSeed(self, seed):
        np.random.seed(seed)
