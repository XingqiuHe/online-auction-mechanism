import numpy as np
import copy
from Parameter import Parameter
from OnlineAlgorithm import OnlineAlgorithm
from OfflineAlgorithm import OfflineAlgorithm
from BenchmarkAlgorithm import BenchmarkAlgorithm


class Algorithm(object):
    def __init__(self, param: Parameter):
        super().__init__()
        # 每个算法的param需要使用alg.offline.param来获取
        self.online = OnlineAlgorithm(param=copy.deepcopy(param))
        self.offline = OfflineAlgorithm(param=copy.deepcopy(param))
        self.benchmark = BenchmarkAlgorithm(param=copy.deepcopy(param))

    # 重置参数
    def resetParam(self, param: Parameter):
        self.online = OnlineAlgorithm(param=copy.deepcopy(param))
        self.offline = OfflineAlgorithm(param=copy.deepcopy(param))
        self.benchmark = BenchmarkAlgorithm(param=copy.deepcopy(param))