import pandas as pd
import numpy as np
from strategy.utils.utils import *
from datetime import datetime
import datetime
from dateutil import rrule
import matplotlib.pyplot as plt
import random
# # typelist=['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'SS', 'BU', 'EG', 'FG', 'FU', 'L', 'MA',
# #           'PP', 'RU', 'SC', 'SP', 'TA', 'V', 'EB', 'LU', 'NR', 'PF', 'PG', 'SA', 'A', 'C', 'CF', 'M', 'OI',
# #           'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 'LC',
# #           'SI', 'SH', 'PX', 'BR', 'AO']
# # for t in typelist:
# #     future_data = pd.read_excel("data/"+t+"_daily.xlsx")
# #     future_data["profit"]=(future_data["close"]-future_data["prev_close"])/future_data["prev_close"]
    
# #     for sig in [5,20,40,252,126,63]:
# #         future_data["sigma"+str(sig)]=future_data['profit'].rolling(window=sig,min_periods=1).std().shift(fill_value=0)
# #     future_data.to_excel("data/"+t+"_daily.xlsx")
        
# # future_data = pd.read_excel("data/"+"AU"+"_daily.xlsx")        
# # print(future_data[["close","sigma20","sigma40","sigma63","sigma126","sigma252"]].corr() )       
        
# # print(a[a["A"]==5])
# # a:pd.DataFrame = pd.read_excel("data/CU_daily.xlsx")
# # print(a.dtypes)
# # print(a[a["date"]>=datetime.datetime.strptime("20100109","%Y%m%d")].index[0])
# # print(a)
# # print(a[a["date"]<datetime.datetime.strptime("20120109","%Y%m%d")][a["date"]>datetime.datetime.strptime("20110509","%Y%m%d")][:1])
# # a=a._append(a[a["date"]<datetime.datetime.strptime("20120109","%Y%m%d")][a["date"]>datetime.datetime.strptime("20110509","%Y%m%d")][:1])
# # print(a)
# # a["close"]=a["close"].apply(lambda a:a*10000)
# # print(a)
# # a="CU_short"
# # b=a.split("_")
# # print(b)
# # date1=datetime.datetime(1991,1,2)
# # date2=datetime.datetime(1992,1,1)
# # print((date2-date1).days)
# # a=[1,2,3]
# # b=[0]
# # c=b.extend(a)
# # print(b)
# # test = ["a","b","c","d","e"]
# # remove = ["b","d"]
# # for i in test:
# #     if i in remove:
# #         test.remove(i)
        
# # print(test)

# # test={"1":lambda x:x+1,"2":lambda x:x*x}
# # # print(test["2"](10))
# # import pandas as pd

# # # 创建示例数据框
# # data = {'x': [1, 2, 3, 4, 5], 'y': [5, 4, 3, 2, 1]}
# # df = pd.DataFrame(data)

# # # 创建rolling对象并计算滚动相关系数
# # rolling_corr = df['x'].rolling(window=3,min_periods=1).mean()
# # print(rolling_corr)
# # vol_bottom_data= pd.read_excel("data/factors/Back_volbottomtargetvol_S20_T0.25_day62.xlsx")
# # vol_top_data=pd.read_excel("data/factors/Back_voltoptargetvol_S5_T0.15_day62.xlsx")
# # for i in [20,40,63,126,252]:
# #     vol_bottom_data["profit"+str(i)]=(vol_bottom_data["volbottomtargetvol_S20_T0.25_day62"].shift(1)-vol_bottom_data["volbottomtargetvol_S20_T0.25_day62"].shift(1+i))/vol_bottom_data["volbottomtargetvol_S20_T0.25_day62"].shift(1+i)
# #     vol_bottom_data["profit"+str(i)] = vol_bottom_data["profit"+str(i)].fillna(0)
    
# #     vol_top_data["profit"+str(i)]=(vol_top_data["voltoptargetvol_S5_T0.15_day62"].shift(1)-vol_top_data["voltoptargetvol_S5_T0.15_day62"].shift(1+i))/vol_top_data["voltoptargetvol_S5_T0.15_day62"].shift(1+i)
# #     vol_top_data["profit"+str(i)]=vol_top_data["profit"+str(i)].fillna(0)

# # print(vol_bottom_data)
# # print(vol_top_data)
# # vol_bottom_data.to_excel("data/factors/Back_volbottomtargetvol_S20_T0.25_day62.xlsx")
# # vol_top_data.to_excel("data/factors/Back_voltoptargetvol_S5_T0.15_day62.xlsx")
# typelist=['AU', 'AG', 'HC', 'I', 'J', 'JM', 'RB', 'SF', 'SM', 'SS', 'BU', 'EG', 'FG', 'FU', 'L', 'MA',
#           'PP', 'RU', 'SC', 'SP', 'TA', 'V', 'EB', 'LU', 'NR', 'PF', 'PG', 'SA', 'A', 'C', 'CF', 'M', 'OI',
#           'RM', 'SR', 'Y', 'JD', 'CS', 'B', 'P', 'LH', 'PK', 'AL', 'CU', 'NI', 'PB', 'SN', 'ZN', 'LC',
#           'SI', 'SH', 'PX', 'BR', 'AO']    
# # for i in typelist:
    
# #     a:pd.DataFrame = pd.read_excel(f"data/{i}_daily.xlsx")
# #     a.iloc[:,1:].to_excel(f"data/{i}_daily.xlsx",index=False)
    
    
# a=pd.DataFrame({"A":[1,23,4],"B":[4,2,1],"C":[64,2,4]})


# print(a["B"].index)


# a=0.1234544123
# print(f"{a:.2f}")

# 定义迷宫的地图和起点、终点
maze = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
]
start = (1, 1)
end = (5, 6)
# 定义遗传规划算法的参数
POPULATION_SIZE = 100
GENERATION_COUNT = 50
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
# 定义个体的数据结构
class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.calculate_fitness()
    def calculate_fitness(self):
        x, y = start
        for gene in self.chromosome:
            if gene == 0:  # 向上移动
                x -= 1
            elif gene == 1:  # 向下移动
                x += 1
            elif gene == 2:  # 向左移动
                y -= 1
            elif gene == 3:  # 向右移动
                y += 1
            if (x, y) == end:
                return 1
            if maze[x][y] == 1:
                return 0
        return 0
# 初始化种群
def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        chromosome = [random.randint(0, 3) for _ in range(50)]  # 假设染色体长度为50
        individual = Individual(chromosome)
        population.append(individual)
    return population
# 选择操作
def selection(population):
    # 使用轮盘赌选择算法
    total_fitness = sum(individual.fitness for individual in population)
    probabilities = [individual.fitness / total_fitness for individual in population]
    selected_individuals = random.choices(population, probabilities, k=POPULATION_SIZE)
    return selected_individuals
# 交叉操作
def crossover(parent1, parent2):
    if random.random() < CROSSOVER_RATE:
        crossover_point = random.randint(1, len(parent1.chromosome) - 1)
        child1_chromosome = parent1.chromosome[:crossover_point] + parent2.chromosome[crossover_point:]
        child2_chromosome = parent2.chromosome[:crossover_point] + parent1.chromosome[crossover_point:]
        child1 = Individual(child1_chromosome)
        child2 = Individual(child2_chromosome)
        return child1, child2
    else:
        return parent1, parent2
# 变异操作
def mutation(individual):
    mutated_chromosome = individual.chromosome.copy()
    for i in range(len(mutated_chromosome)):
        if random.random() < MUTATION_RATE:
            mutated_chromosome[i] = random.randint(0, 3)
    return Individual(mutated_chromosome)
# 主函数
def main():
    population = initialize_population()
    best_fitness = 0
    best_individual = None
    for generation in range(GENERATION_COUNT):
        selected_individuals = selection(population)
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = random.sample(selected_individuals, 2)
            child1, child2 = crossover(parent1, parent2)
            child1 = mutation(child1)
            child2 = mutation(child2)
            new_population.extend([child1, child2])
        population = new_population
        # 更新最佳个体
        for individual in population:
            if individual.fitness > best_fitness:
                best_fitness = individual.fitness
                best_individual = individual
        print("Generation:", generation + 1)
        print("Best Individual:", best_individual.chromosome)
        print("Best Fitness:", best_fitness)
        print()
    # 输出最终结果
    print("Optimal Solution:")
    print("Chromosome:", best_individual.chromosome)
    print("Path:")
    x, y = start
    for gene in best_individual.chromosome:
        if gene == 0:  # 向上移动
            x -= 1
        elif gene == 1:  # 向下移动
            x += 1
        elif gene == 2:  # 向左移动
            y -= 1
        elif gene == 3:  # 向右移动
            y += 1
        print("(", x, ",", y, ")")
        if (x, y) == end:
            break
if __name__ == "__main__":
    main()