from Parameter import Parameter
from Environment import Environment
from Algorithm import Algorithm
from Plot import Plot
import numpy as np
import time

N = 1500
M = 15
T = 20

env_mode = 0   # mode 0 普通 1 突发 2 垃圾优先 3 垃圾优先+突发
if env_mode == 0:
    mode_str = '普通数据模式'
elif env_mode == 1:
    mode_str = '突发模式'
elif env_mode == 2:
    mode_str = '垃圾优先模式'
else:
    mode_str = '垃圾优先+突发模式'

max_seed = 5

def figure_1():
    N_ = 3000
    out = np.zeros([5, N_])  # 1. 在线算法run-time performance。折线图。纵坐标是time-average social welfare，横坐标time slots，每一条线代表一个在线算法。
    param = Parameter(N=N_, M=M, T=T)
    env = Environment(param=param)

    # env.generateMap(xSize=500, ySize=500)
    # env.saveMap(filename='MAP_' + str(param.M) + '_' + str(param.N) + '.npz')
    # env.printMap(xSize=500, ySize=500, filename='MAP_' + str(param.M) + '_' + str(param.N) + '.txt')
    # env.loadMap(filename='MAP_' + str(param.M) + '_' + str(param.N) + '.npz')

    for seed in range(max_seed):
        param.setSeed(seed)
        env.setSeed(seed)
        env.generateMap(xSize=500, ySize=500)
        env.generateAPs()
        env.generateTasks(mode=env_mode)  # mode 0 普通 1 突发 2 垃圾优先 3 垃圾优先+突发

        alg = Algorithm(param=param)
        alg.online.setSeed(seed)
        alg.benchmark.setSeed(seed)

        logs = [[i for i in range(N_)], alg.online.execute()]
        logs.extend(alg.benchmark.runBenchmark())
        out += np.array(logs)

    out = out.T/max_seed
    out_name = np.array([['Step', 'Proposed_algorithm', 'Random_algorithm', 'Greedy_algorithm', 'FIFO_algorithm']])
    out = np.append(out_name, out, axis=0)

    print(out.shape)
    np.savetxt(f"output/{mode_str}/logs_socialwelfare_{N_}.csv", out, fmt='%s', delimiter=',')

def figure_2_4():
    min_N = 1000
    max_N = 6000
    step_N = 1000
    out2 = np.zeros([5, int((max_N-min_N)/step_N)+1]) # 2. 在线算法social welfare对比。bar图。纵坐标social welfare，横坐标Number of bids/users。每一个bar代表一个在线算法。
    out4 = np.zeros([5, int((max_N-min_N)/step_N)+1]) # 4. 在线算法用户满意度对比。bar图。纵坐标用户满意度（percentage of winners）， 横坐标用户数量。每一个bar代表一个在线算法。
    out6 = np.zeros([5, int((max_N-min_N)/step_N)+1]) # 6. 运行时间表。三个在线算法在不同参数下的运行时间。这个不用画图，只用把数据记录下来即可。如有必要我们列个表格就行。
    index = 0
    for N_ in range(min_N, max_N+1, step_N):
        print(f'\nN={N_}')
        param = Parameter(N=N_, M=M, T=T)
        env = Environment(param=param)

        for seed in range(max_seed):
            param.setSeed(seed)
            env.setSeed(seed)
            env.generateMap(xSize=500, ySize=500)
            env.generateAPs()
            env.generateTasks(mode=env_mode)

            alg = Algorithm(param=param)
            alg.online.setSeed(seed)
            alg.benchmark.setSeed(seed)

            alg.online.execute()
            alg.benchmark.runBenchmark()

            out2[1][index] += alg.online.final_social_welfare
            out2[2][index] += alg.benchmark.final_social_welfare[0]
            out2[3][index] += alg.benchmark.final_social_welfare[1]
            out2[4][index] += alg.benchmark.final_social_welfare[2]

            out4[1][index] += alg.online.final_user_satisfaction
            out4[2][index] += alg.benchmark.final_user_satisfaction[0]
            out4[3][index] += alg.benchmark.final_user_satisfaction[1]
            out4[4][index] += alg.benchmark.final_user_satisfaction[2]

            out6[1][index] += alg.online.execute_time
            out6[2][index] += alg.benchmark.execute_time[0]
            out6[3][index] += alg.benchmark.execute_time[1]
            out6[4][index] += alg.benchmark.execute_time[2]

        out2[0][index] = N_
        out2[1][index] /= max_seed
        out2[2][index] /= max_seed
        out2[3][index] /= max_seed
        out2[4][index] /= max_seed

        out4[0][index] = N_
        out4[1][index] /= max_seed
        out4[2][index] /= max_seed
        out4[3][index] /= max_seed
        out4[4][index] /= max_seed

        out6[0][index] = N_
        out6[1][index] /= max_seed
        out6[2][index] /= max_seed
        out6[3][index] /= max_seed
        out6[4][index] /= max_seed

        index += 1

    out2 = out2.T
    out_name = np.array([['Number of Bids', 'Proposed_algorithm', 'Random_algorithm', 'Greedy_algorithm', 'FIFO_algorithm']])
    out2 = np.append(out_name, out2, axis=0)
    print(out2.shape)
    np.savetxt(f"output/{mode_str}/logs_Bids_sw.csv", out2, fmt='%s', delimiter=',')

    out4 = out4.T
    out4 = np.append(out_name, out4, axis=0)
    print(out4.shape)
    np.savetxt(f"output/{mode_str}/logs_Bids_pow.csv", out4, fmt='%s', delimiter=',')

    out6 = out6.T
    out6 = np.append(out_name, out6, axis=0)
    print(out6.shape)
    np.savetxt(f"output/{mode_str}/logs_Bids_time.csv", out6, fmt='%s', delimiter=',')


def figure_3():
    min_M = 0
    max_M = 20
    step_M = 5
    out3 = np.zeros([5, int((max_M-min_M)/step_M)+1]) # 3. 在线算法social welfare对比。bar图。纵坐标social welfare，横坐标Number of BSs。每一个bar代表一个在线算法。
    out6 = np.zeros([5, int((max_M - min_M) / step_M)+1])  # 6. 运行时间表。三个在线算法在不同参数下的运行时间。这个不用画图，只用把数据记录下来即可。如有必要我们列个表格就行。
    index = 0
    for M_ in range(min_M, max_M+1, step_M):
        if M_ == 0:
            M_ = 1
        print(f'\nM={M_}')
        param = Parameter(N=N, M=M_, T=T)
        env = Environment(param=param)

        for seed in range(max_seed):
            param.setSeed(seed)
            env.setSeed(seed)
            env.generateMap(xSize=500, ySize=500)
            env.generateAPs()
            env.generateTasks(mode=env_mode)

            alg = Algorithm(param=param)
            alg.online.setSeed(seed)
            alg.benchmark.setSeed(seed)

            alg.online.execute()
            alg.benchmark.runBenchmark()

            out3[1][index] += alg.online.final_social_welfare
            out3[2][index] += alg.benchmark.final_social_welfare[0]
            out3[3][index] += alg.benchmark.final_social_welfare[1]
            out3[4][index] += alg.benchmark.final_social_welfare[2]

            out6[1][index] += alg.online.execute_time
            out6[2][index] += alg.benchmark.execute_time[0]
            out6[3][index] += alg.benchmark.execute_time[1]
            out6[4][index] += alg.benchmark.execute_time[2]

        out3[0][index] = M_
        out3[1][index] /= max_seed
        out3[2][index] /= max_seed
        out3[3][index] /= max_seed
        out3[4][index] /= max_seed

        out6[0][index] = M_
        out6[1][index] /= max_seed
        out6[2][index] /= max_seed
        out6[3][index] /= max_seed
        out6[4][index] /= max_seed

        index += 1

    out3 = out3.T
    out_name = np.array([['Number of BSs', 'Proposed_algorithm', 'Random_algorithm', 'Greedy_algorithm', 'FIFO_algorithm']])
    out3 = np.append(out_name, out3, axis=0)
    print(out3.shape)
    np.savetxt(f"output/{mode_str}/logs_BS_sw.csv", out3, fmt='%s', delimiter=',')

    out6 = out6.T
    out6 = np.append(out_name, out6, axis=0)
    print(out6.shape)
    np.savetxt(f"output/{mode_str}/logs_BSs_time.csv", out6, fmt='%s', delimiter=',')


def figure_5():
    min_UL = 1
    max_UL = 10
    step_UL = 2
    out5 = np.zeros([int((max_UL-min_UL)/step_UL)+1, N+1]) # 5. 本算法在不同U/L下的表箱。折线图。纵坐标time-average social welfare，横坐标time slots。每一条线代表不同的U/L。
    out6 = np.zeros([2, int((max_UL - min_UL) / step_UL)+1])  # 6. 运行时间表。三个在线算法在不同参数下的运行时间。这个不用画图，只用把数据记录下来即可。如有必要我们列个表格就行。

    param = Parameter(N=N, M=M, T=T)
    env = Environment(param=param)

    index = 0
    for UL_ in range(min_UL, max_UL+1, step_UL):
        if UL_ == 0:
            UL_ = 1
        print(f'\nUL={UL_}')
        for seed in range(max_seed):
            param.setSeed(seed)
            env.setSeed(seed)
            env.generateMap(xSize=500, ySize=500)
            env.generateAPs()
            env.generateTasks(mode=env_mode)

            alg = Algorithm(param=param)
            alg.online.setSeed(seed)
            alg.online.set_UL(UL=UL_)

            logs = alg.online.execute()

            out5[index][1:] += np.array(logs)

            out6[1][index] += alg.online.execute_time

        out5[index] /= max_seed
        out5[index][0] = UL_

        out6[0][index] = UL_
        out6[1][index] /= max_seed

        index += 1

    out_name = [i for i in range(-1, N)]
    out_name[0] = 'Step'
    out_name = np.array([out_name])
    out5 = np.append(out_name, out5, axis=0).T
    print(out5.shape)
    np.savetxt(f"output/{mode_str}/logs_UL.csv", out5, fmt='%s', delimiter=',')

    out6 = out6.T
    out_name = np.array([['UL', 'Proposed_algorithm']])
    out6 = np.append(out_name, out6, axis=0)
    print(out6.shape)
    np.savetxt(f"output/{mode_str}/logs_UL_time.csv", out6, fmt='%s', delimiter=',')

###############################################新图###########################################################
# 6、7: 新增两个profits（纵坐标）对比任务数量和基站数量（两个横坐标）的图。基站数量的范围是5,10,15,20,25，任务的数量范围是500,1000,1500,2000,2500。默认基站数量是15，默认任务数量是1500
# 8、9: 展示基站之间由于合作带来的额外收益。每个基站固定连100个task，比如如果只有一个基站，那么任务也只有100个。基站数量范围是1,5,10,15,20，三个图纵坐标分别为平均sw（就是总sw除以基站数目），资源利用率（三个bar分别代表s，f，w），还有用户满意度。平均sw和用户满意度四个bar依旧对应四个不同算法，资源利用率只用跑proposed

def figure_6():
    # profits 对比 任务数量
    min_N = 500
    max_N = 2500
    step_N = 500
    out7 = np.zeros([5, int((max_N - min_N) / step_N) + 1])  # 7. profits
    out6 = np.zeros([5, int((max_N - min_N) / step_N) + 1])  # 6. 运行时间表。三个在线算法在不同参数下的运行时间。这个不用画图，只用把数据记录下来即可。如有必要我们列个表格就行。
    index = 0
    for N_ in range(min_N, max_N + 1, step_N):
        print(f'\nN={N_}')
        param = Parameter(N=N_, M=M, T=T)
        env = Environment(param=param)

        for seed in range(max_seed):
            param.setSeed(seed)
            env.setSeed(seed)
            env.generateMap(xSize=500, ySize=500)
            env.generateAPs()
            env.generateTasks(mode=env_mode)

            alg = Algorithm(param=param)
            alg.online.setSeed(seed)
            alg.benchmark.setSeed(seed)

            alg.online.execute()
            alg.benchmark.runBenchmark()

            out7[1][index] += alg.online.param.getProfits()
            out7[2][index] += alg.benchmark.randomParam.getProfits()
            out7[3][index] += alg.benchmark.greedyParam.getProfits()
            out7[4][index] += alg.benchmark.FIFOParam.getProfits()

            out6[1][index] += alg.online.execute_time
            out6[2][index] += alg.benchmark.execute_time[0]
            out6[3][index] += alg.benchmark.execute_time[1]
            out6[4][index] += alg.benchmark.execute_time[2]

        out7[0][index] = N_
        out7[1][index] /= max_seed
        out7[2][index] /= max_seed
        out7[3][index] /= max_seed
        out7[4][index] /= max_seed

        out6[0][index] = N_
        out6[1][index] /= max_seed
        out6[2][index] /= max_seed
        out6[3][index] /= max_seed
        out6[4][index] /= max_seed

        index += 1

    out7 = out7.T
    out_name = np.array([['Number of Bids', 'Proposed_algorithm', 'Random_algorithm', 'Greedy_algorithm', 'FIFO_algorithm']])
    out7 = np.append(out_name, out7, axis=0)
    np.savetxt(f"output/{mode_str}/new1_profits_Bids.csv", out7, fmt='%s', delimiter=',')

    out6 = out6.T
    out6 = np.append(out_name, out6, axis=0)
    np.savetxt(f"output/{mode_str}/new1_profits_Bids_time.csv", out6, fmt='%s', delimiter=',')

def figure_7():
    # profits 对比 基站数量
    min_M = 5
    max_M = 25
    step_M = 5
    out7 = np.zeros([5, int((max_M - min_M) / step_M) + 1])  # 7. profits
    out6 = np.zeros([5, int((max_M - min_M) / step_M) + 1])  # 6. 运行时间表
    index = 0
    for M_ in range(min_M, max_M + 1, step_M):
        if M_ == 0:
            M_ = 1
        print(f'\nM={M_}')
        param = Parameter(N=N, M=M_, T=T)
        env = Environment(param=param)

        for seed in range(max_seed):
            param.setSeed(seed)
            env.setSeed(seed)
            env.generateMap(xSize=500, ySize=500)
            env.generateAPs()
            env.generateTasks(mode=env_mode)

            alg = Algorithm(param=param)
            alg.online.setSeed(seed)
            alg.benchmark.setSeed(seed)

            alg.online.execute()
            alg.benchmark.runBenchmark()

            out7[1][index] += alg.online.param.getProfits()
            out7[2][index] += alg.benchmark.randomParam.getProfits()
            out7[3][index] += alg.benchmark.greedyParam.getProfits()
            out7[4][index] += alg.benchmark.FIFOParam.getProfits()

            out6[1][index] += alg.online.execute_time
            out6[2][index] += alg.benchmark.execute_time[0]
            out6[3][index] += alg.benchmark.execute_time[1]
            out6[4][index] += alg.benchmark.execute_time[2]

        out7[0][index] = M_
        out7[1][index] /= max_seed
        out7[2][index] /= max_seed
        out7[3][index] /= max_seed
        out7[4][index] /= max_seed

        out6[0][index] = M_
        out6[1][index] /= max_seed
        out6[2][index] /= max_seed
        out6[3][index] /= max_seed
        out6[4][index] /= max_seed

        index += 1

    out7 = out7.T
    out_name = np.array(
        [['Number of BSs', 'Proposed_algorithm', 'Random_algorithm', 'Greedy_algorithm', 'FIFO_algorithm']])
    out7 = np.append(out_name, out7, axis=0)
    np.savetxt(f"output/{mode_str}/new1_profits_BS.csv", out7, fmt='%s', delimiter=',')

    out6 = out6.T
    out6 = np.append(out_name, out6, axis=0)
    np.savetxt(f"output/{mode_str}/new1_profits_BS_time.csv", out6, fmt='%s', delimiter=',')

def figure_8():
    # 平均sw、资源利用率、用户满意度 对比 任务数量
    min_N = 500
    max_N = 2500
    step_N = 500
    out8 = np.zeros([5, int((max_N - min_N) / step_N) + 1])  # 8. 平均sw=总sw/BS数量
    out9 = np.zeros([5, int((max_N - min_N) / step_N) + 1])  # 9. cpu资源利用率
    out4 = np.zeros([5, int((max_N - min_N) / step_N) + 1])  # 4. 用户满意度
    out6 = np.zeros([5, int((max_N - min_N) / step_N) + 1])  # 6. 运行时间表
    index = 0
    for N_ in range(min_N, max_N + 1, step_N):
        if N_ == 0:
            N_ = 1
        print(f'\nN={N_}')
        param = Parameter(N=N_, M=M, T=T)
        env = Environment(param=param)

        for seed in range(max_seed):
            param.setSeed(seed)
            env.setSeed(seed)
            env.generateMap(xSize=500, ySize=500)
            env.generateAPs()
            env.generateTasks(mode=env_mode)

            alg = Algorithm(param=param)
            alg.online.setSeed(seed)
            alg.benchmark.setSeed(seed)

            alg.online.execute()
            alg.benchmark.runBenchmark()

            out8[1][index] += alg.online.final_social_welfare / M
            out8[2][index] += alg.benchmark.final_social_welfare[0] / M
            out8[3][index] += alg.benchmark.final_social_welfare[1] / M
            out8[4][index] += alg.benchmark.final_social_welfare[2] / M

            out4[1][index] += alg.online.final_user_satisfaction
            out4[2][index] += alg.benchmark.final_user_satisfaction[0]
            out4[3][index] += alg.benchmark.final_user_satisfaction[1]
            out4[4][index] += alg.benchmark.final_user_satisfaction[2]

            out6[1][index] += alg.online.execute_time
            out6[2][index] += alg.benchmark.execute_time[0]
            out6[3][index] += alg.benchmark.execute_time[1]
            out6[4][index] += alg.benchmark.execute_time[2]

            k = 1
            alg_id = 1
            for p in [alg.online.param, alg.benchmark.randomParam, alg.benchmark.greedyParam, alg.benchmark.FIFOParam]:
                usage = 0.
                total = 0.
                for j in range(p.M):
                    total += p.C[k][j] * p.T
                    for t in range(p.T):
                        usage += p.z[k][j][t]
                out9[alg_id][index] += usage / total
                alg_id += 1

        out8[0][index] = N_
        out8[1][index] /= max_seed
        out8[2][index] /= max_seed
        out8[3][index] /= max_seed
        out8[4][index] /= max_seed

        out9[0][index] = N_
        out9[1][index] /= max_seed
        out9[2][index] /= max_seed
        out9[3][index] /= max_seed
        out9[4][index] /= max_seed

        out4[0][index] = N_
        out4[1][index] /= max_seed
        out4[2][index] /= max_seed
        out4[3][index] /= max_seed
        out4[4][index] /= max_seed

        out6[0][index] = N_
        out6[1][index] /= max_seed
        out6[2][index] /= max_seed
        out6[3][index] /= max_seed
        out6[4][index] /= max_seed

        index += 1

    out8 = out8.T
    out_name = np.array(
        [['Number of Bids', 'Proposed_algorithm', 'Random_algorithm', 'Greedy_algorithm', 'FIFO_algorithm']])
    out8 = np.append(out_name, out8, axis=0)
    np.savetxt(f"output/{mode_str}/new2_sw_avg_Bids.csv", out8, fmt='%s', delimiter=',')

    out4 = out4.T
    out4 = np.append(out_name, out4, axis=0)
    np.savetxt(f"output/{mode_str}/new2_satisfy_Bids.csv", out4, fmt='%s', delimiter=',')

    out6 = out6.T
    out6 = np.append(out_name, out6, axis=0)
    np.savetxt(f"output/{mode_str}/new2_Bids_time.csv", out6, fmt='%s', delimiter=',')

    out9 = out9.T
    out9 = np.append(out_name, out9, axis=0)
    np.savetxt(f"output/{mode_str}/new2_usage_Bids.csv", out9, fmt='%s', delimiter=',')

def figure_9():
    # 平均sw、资源利用率、用户满意度 对比 基站数量
    min_M = 5
    max_M = 25
    step_M = 5
    out8 = np.zeros([5, int((max_M - min_M) / step_M) + 1])  # 8. 平均sw=总sw/BS数量
    out9 = np.zeros([5, int((max_M - min_M) / step_M) + 1])  # 9. cpu资源利用率
    out4 = np.zeros([5, int((max_M - min_M) / step_M) + 1])  # 4. 用户满意度
    out6 = np.zeros([5, int((max_M - min_M) / step_M) + 1])  # 6. 运行时间表
    index = 0
    for M_ in range(min_M, max_M + 1, step_M):
        if M_ == 0:
            M_ = 1
        print(f'\nM={M_}')
        param = Parameter(N=N, M=M_, T=T)
        env = Environment(param=param)

        for seed in range(max_seed):
            param.setSeed(seed)
            env.setSeed(seed)
            env.generateMap(xSize=500, ySize=500)
            env.generateAPs()
            env.generateTasks(mode=env_mode)

            alg = Algorithm(param=param)
            alg.online.setSeed(seed)
            alg.benchmark.setSeed(seed)

            alg.online.execute()
            alg.benchmark.runBenchmark()

            out8[1][index] += alg.online.final_social_welfare / M_
            out8[2][index] += alg.benchmark.final_social_welfare[0] / M_
            out8[3][index] += alg.benchmark.final_social_welfare[1] / M_
            out8[4][index] += alg.benchmark.final_social_welfare[2] / M_

            out4[1][index] += alg.online.final_user_satisfaction
            out4[2][index] += alg.benchmark.final_user_satisfaction[0]
            out4[3][index] += alg.benchmark.final_user_satisfaction[1]
            out4[4][index] += alg.benchmark.final_user_satisfaction[2]

            out6[1][index] += alg.online.execute_time
            out6[2][index] += alg.benchmark.execute_time[0]
            out6[3][index] += alg.benchmark.execute_time[1]
            out6[4][index] += alg.benchmark.execute_time[2]

            k = 1
            algid = 1
            for p in [alg.online.param, alg.benchmark.randomParam, alg.benchmark.greedyParam, alg.benchmark.FIFOParam]:
                usage = 0.
                total = 0.
                for j in range(p.M):
                    total += p.C[k][j] * p.T
                    for t in range(p.T):
                        usage += p.z[k][j][t]
                out9[algid][index] += usage / total
                algid += 1

        out8[0][index] = M_
        out8[1][index] /= max_seed
        out8[2][index] /= max_seed
        out8[3][index] /= max_seed
        out8[4][index] /= max_seed

        out9[0][index] = M_
        out9[1][index] /= max_seed
        out9[2][index] /= max_seed
        out9[3][index] /= max_seed
        out9[4][index] /= max_seed

        out4[0][index] = M_
        out4[1][index] /= max_seed
        out4[2][index] /= max_seed
        out4[3][index] /= max_seed
        out4[4][index] /= max_seed

        out6[0][index] = M_
        out6[1][index] /= max_seed
        out6[2][index] /= max_seed
        out6[3][index] /= max_seed
        out6[4][index] /= max_seed

        index += 1

    out8 = out8.T
    out_name = np.array(
        [['Number of BSs', 'Proposed_algorithm', 'Random_algorithm', 'Greedy_algorithm', 'FIFO_algorithm']])
    out8 = np.append(out_name, out8, axis=0)
    np.savetxt(f"output/{mode_str}/new2_sw_avg_BS.csv", out8, fmt='%s', delimiter=',')

    out4 = out4.T
    out4 = np.append(out_name, out4, axis=0)
    np.savetxt(f"output/{mode_str}/new2_satisfy_BS.csv", out4, fmt='%s', delimiter=',')

    out6 = out6.T
    out6 = np.append(out_name, out6, axis=0)
    np.savetxt(f"output/{mode_str}/new2_BS_time.csv", out6, fmt='%s', delimiter=',')

    out9 = out9.T
    out9 = np.append(out_name, out9, axis=0)
    np.savetxt(f"output/{mode_str}/new2_usage_BS.csv", out9, fmt='%s', delimiter=',')

# for env_mode in range(3):
#     if env_mode == 0:
#         mode_str = '普通数据模式'
#     elif env_mode == 1:
#         mode_str = '突发模式'
#     elif env_mode == 2:
#         mode_str = '垃圾优先模式'
#     else:
#         mode_str = '垃圾优先+突发模式'
#     print(mode_str)
#     print('============1================')
#     figure_1()
#     print('============2&4================')
#     figure_2_4()
#     print('============3================')
#     figure_3()
#     # print('============5================')
#     # figure_5()
#
figure_6()
figure_7()
# figure_8()
# figure_9()

# N = 10
# M = 1
# T = 5
# param = Parameter(N=N, M=M, T=T)
# env = Environment(param=param)
# env.generateMap(xSize=500, ySize=500)
# env.generateAPs()
# env.generateTasks(mode=0)  # mode 0 普通 1 突发 2 垃圾优先 3 垃圾优先+突发
# alg = Algorithm(param=param)
# alg.online.execute()
# alg.benchmark.runBenchmark()