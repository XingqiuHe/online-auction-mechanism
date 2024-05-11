import numpy as np
from Parameter import Parameter
# np.random.seed(1190)


class Environment(object):
    def __init__(self, param: Parameter):
        super().__init__()
        self.param = param

    '''生成基站信息'''
    def generateAPs(self):
        M = self.param.M
        self.param.U[0] = 50. / 1e6     # 1Hz带宽的价格上限
        self.param.U[1] = 50. / 1e9     # 1Hz频率的价格上限
        self.param.U[2] = 50. / 1e6     # 1Byte存储空间的价格上限
        self.param.L[0] = 1. / 1e6      # 1Hz带宽的价格下限
        self.param.L[1] = 1. / 1e9      # 1Hz频率的价格下限
        self.param.L[2] = 1. / 1e6      # 1Byte存储空间的价格下限
        self.param.L_w = 1. / 1e6       # 1Hz带宽的价格下限
        self.param.U_w = 50. / 1e6      # 1Hz带宽的价格上限
        for j in range(M):
            self.param.sigma_2[j] = 1e-9
            self.param.beta[1][j] = np.random.randint(4, 6) / 10.
            self.param.beta[2][j] = np.random.randint(5, 20) / 1000.
            self.param.gamma[1][j] = np.random.randint(17, 22) / 10.
            self.param.gamma[2][j] = np.random.randint(5, 10) / 10.
            self.param.C[0][j] = 40. * 1e6  # 40MHz
            self.param.C[1][j] = 5. * 1e9  # 5GHz
            self.param.C[2][j] = 5. * 1e9  # 5GB

    '''生成任务信息
    :parameter mode: 生成任务的模式
    mode 0: 普通模式。在[0, T - 3]的范围内生成IID的任务
    mode 1: 突发任务模式。将T以10为单位分成多份，[10*k, 10*k + 9]为一份，其中每个slot上任务的数量为26:26:1:1:1:1:1:1:1:1
    mode 2: 垃圾任务优先模式。将普通模式的任务排序，将价值低的任务放在前面
    mode 3: 垃圾任务优先+突发模式。
    '''
    def generateTasks(self, mode: 0):
        N = self.param.N
        T = self.param.T

        segLen = 10     # 一个时隙段的长度
        segNum = int(T / segLen)  # 获取时间段T可以分成多少段，每一段长10slots，最后一段除外

        if T <= 3:
            print("设定的T太小!")
            raise ValueError
        tasks_num_in_slots = [0 for _ in range(T)]
        now_num = [0 for _ in range(T)]
        if mode == 0 or mode == 2:
            while np.sum(tasks_num_in_slots) < self.param.N:
                tasks_num_in_slots = np.random.poisson(np.ceil(1.0*self.param.N/self.param.T), self.param.T)
        elif mode == 1 or mode == 3:
            # 用poison为每个时间段分配任务数
            tasks_num_in_groups = [0 for _ in range(segNum)]
            while np.sum(tasks_num_in_groups) < self.param.N:
                tasks_num_in_groups = np.random.poisson(np.ceil(1.0*self.param.N/segNum), segNum)
            # print(tasks_num_in_groups)
            # 在每个时间段中，为slots分配任务数
            for k in range(segNum):
                for q in range(10):
                    index = 10*k+q
                    if index >= self.param.T:
                        break
                    if q < 2:
                        tasks_num_in_slots[index] = np.ceil(tasks_num_in_groups[k] * 26 / 60)
                        # tasks_num_in_slots[index] = np.ceil(tasks_num_in_groups[k] * 96. / 200.)
                    else:
                        tasks_num_in_slots[index] = np.ceil(tasks_num_in_groups[k] * 1 / 60)
                        # tasks_num_in_slots[index] = np.ceil(tasks_num_in_groups[k] * 1. / 200.)
        # print(tasks_num_in_slots)
        for i in range(N):
            self.param.s[i] = np.random.randint(10, 30) * 1e6      # 转换成Byte单位
            self.param.w[i] = int(100 * self.param.s[i])
            self.param.P[i] = 1.5

            # if mode == 0 or mode == 2:
            #     self.param.t[i] = np.random.randint(0, T - 3)
            # elif mode == 1 or mode == 3:
            #     if T % segLen < 3:
            #         segNum -= 1       # 至少要保证t[i] + 3 < T，因为t_c_1 >= t_t_0 + 2
            #     k = np.random.randint(0, segNum)  # 选择当前任务的t[i]将落到哪一段
            #     self.param.t[i] = np.random.randint(segLen*k, segLen*k+3)

            target_t = np.random.randint(0, T)
            counter = 0
            add_num = 1 if np.random.randint(0, 2) == 1 else -1
            while tasks_num_in_slots[target_t] <= now_num[target_t] + 1 or target_t >= T-3:
                target_t += add_num
                if target_t >= T-3:
                    target_t = 0
                if target_t < 0:
                    target_t = T-3 - 1
                counter += 1
                if counter == T-3:
                    break
            now_num[target_t] += 1
            self.param.t[i] = target_t

            self.create_b(i)

        if mode == 2 or mode == 3:
            def getMaximum(elem):
                return elem(0)
            self.param.b.sort(key=getMaximum)
        # print(now_num)

    # lambda记录的是作用域，而不是对象。而py的for是一个作用域，最后lambda全部都是记录的for最终状态的i值。所以应该借用一层函数对lambda的赋值进行包装
    def create_b(self, i):
        mu = 7
        interval = 6
        b_1 = float(np.random.randint((mu - interval) * 10, (mu + interval) * 10 - 1) * 1.)
        b_2 = float(np.random.randint(1, 10) * 1.)
        self.param.b[i] = lambda x: b_1 - b_2 * x

    '''生成AP与UE的分布地图
    :param xSize: 地图的横轴大小（m）
    :param ySize: 地图的纵轴大小（m）
    :param internal: 采样点之间的间距（m）
    '''
    def generateMap(self, xSize=500, ySize=500, internal=1):
        # 只需要保证AP不会在地图上重叠，不同Task可能来自一个发生源
        # Generate APs
        for j in range(self.param.M):
            while 1:
                xj = np.random.randint(0, xSize/internal) * internal
                yj = np.random.randint(0, ySize/internal) * internal
                isExist = False
                for [x, y] in self.param.locAP:
                    if x == xj and y == yj:
                        isExist = True
                        break
                if not isExist:
                    self.param.locAP[j] = [xj, yj]
                    break
        # Generate UEs
        for i in range(self.param.N):
            xi = np.random.randint(0, xSize / internal) * internal
            yi = np.random.randint(0, ySize / internal) * internal
            self.param.locUE[i] = [xi, yi]
        print('Success to generate map!')
        self.calChannelGain()

    '''从文件中读取AP与UE的分布地图'''
    def loadMap(self, filename: str):
        npzfile = np.load('data/'+filename)
        self.param.locAP = npzfile['arr_0']
        self.param.locUE = npzfile['arr_1']
        print('Success to load map!')
        self.calChannelGain()

    '''将AP与UE的分布地图保存到文件'''
    def saveMap(self, filename: str):
        np.savez('data/'+filename, self.param.locAP, self.param.locUE)
        print('Success to save map!')

    '''将分布地图打印到文件中'''
    def printMap(self, filename: str, xSize=500, ySize=500, internal=1):
        # A代表AP U代表UE
        mecMap = [[' ' for _ in range(int(xSize / internal))] for _ in range(int(ySize / internal))]
        for i in range(self.param.N):
            x = int(self.param.locUE[i][0] / internal)
            y = int(self.param.locUE[i][1] / internal)
            mecMap[x][y] = 'U'
        for j in range(self.param.M):
            x = int(self.param.locAP[j][0] / internal)
            y = int(self.param.locAP[j][1] / internal)
            mecMap[x][y] = 'A'
        np.savetxt('data/' + filename, mecMap, fmt='%c')
        print('Success to print map!')

    '''计算信道的上下行增益'''
    def calChannelGain(self):
        theta_U = 6.25 * 10**(-4)
        for i in range(self.param.N):
            xi, yi = self.param.locUE[i]
            for j in range(self.param.M):
                xj, yj = self.param.locAP[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                h_random = np.sqrt(np.random.normal(loc=0, scale=np.sqrt(1/2))**2 + np.random.normal(loc=0, scale=np.sqrt(1/2))**2)
                self.param.h_U[i][j] = theta_U * max(dist, 1e-6)**(-3) * h_random * 125
                # self.param.h_D[i][j] = 2 * self.h_U[i][j]

    def setSeed(self, seed):
        np.random.seed(seed)
