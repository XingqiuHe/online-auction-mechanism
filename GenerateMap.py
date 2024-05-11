from Environment import Environment
from Parameter import Parameter
'''仅用于生成MAP文件'''

for n in range(55, 105, 10):
    param = Parameter(N=n)
    env = Environment(param=param)
    env.generateMap()
    env.saveMap(filename='MAP_' + str(param.M) + '_' + str(param.N) + '.npz')
    env.printMap(filename='MAP_' + str(param.M) + '_' + str(param.N) + '.txt')

for m in range(20, 40, 5):
    param = Parameter(M=m)
    env = Environment(param=param)
    env.generateMap()
    env.saveMap(filename='MAP_' + str(param.M) + '_' + str(param.N) + '.npz')
    env.printMap(filename='MAP_' + str(param.M) + '_' + str(param.N) + '.txt')

