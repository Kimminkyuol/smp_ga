import random

import matplotlib.pyplot as plt
import numpy as np
import pygad


class SMP:
    num_pairs = 0
    num_generations = 100

    solution_per_population = 50

    num_parents_mating = 10
    crossover_probability = 0.1
    mutation_probability = 0.9

    pm, pw = None, None

    initial_population = None

    fitness = np.full((num_generations, 4), 0, dtype=np.float32)
    idx = -1

    def __init__(self, num_pairs, pm, pw):
        self.num_pairs = num_pairs

        self.pm = pm
        self.pw = pw

        self.initial_population = [list(range(self.num_pairs))] * self.num_generations

    def fitness_func(self, _, solution, __):
        """
        최적화 위한 목적 함수 (fitness function)
        안정적 쌍, 행복 총합, 평등 행복 총합의 최대화를 목적으로 함

        Args:
            solution: 현재 세대

        return: 적합도 평가 점수
        """

        hapm = 0  # 남성 선호도 총합
        hapw = 0  # 여성 선호도 총합

        num_stable_pair = 0  # 안정적 쌍의 수

        # 안정적 쌍 수 세기
        for woman, man in enumerate(solution):
            wpm = np.where(self.pm[man] == woman)[0][0]  # 여성에 대한 남성의 선호도 (순서)
            wpw = np.where(self.pw[woman] == man)[0][0]  # 남성에 대한 여성의 선호도 (순서)

            # 순서가 작을 수록 좋음 (1등이 3등보다 좋음)
            hapm += self.num_pairs - wpm
            hapw += self.num_pairs - wpw

            for m in self.pw[woman][:wpw]:  # 여성(1-1)이 현재 파트너로 지정된 남성(2-1)보다 더 선호하는 남성(2-2)의 목록
                w = np.where(solution == m)[0][0]  # 해당 남성(2-2)이 현재 파트너로 지정된 여성(1-2)
                wpmm = np.where(self.pm[m] == w)[0][0]  # 현재 파트너로 지정된 여성(1-2)에 대한 해당 남성의 선호도 (순서)

                if woman in self.pm[m][:wpmm]:  # 해당 남성(2-2)이 현재 파트너보다 선호하는 여성 중에 원래 여성(1-1)이 있다면
                    break  # 안정적 쌍이 아님

            else:  # 모든 경우에 대해 break가 발생하지 않았다면
                num_stable_pair += 1  # 안정적 쌍이 맞음

        hap = (hapm + hapw) / self.num_pairs  # 커플당 총 행복
        ehap = abs(hapm - hapw) / self.num_pairs  # 커플당 평등적(egalitarian) 총 행복 (남녀간 총 행복 행복 차이)

        fitness = num_stable_pair + hap - ehap

        l_fitness = self.fitness[self.idx]
        if l_fitness[3] == 0 or l_fitness[3] < fitness:
            self.fitness[self.idx] = (num_stable_pair, hap, ehap, fitness)

        # 1. 안정적 쌍의 수 최대화 (+)
        # 2. 커플당 총 행복 최대화 (+)
        # 3. 커플당 평등적 총 행복 최소화 (남녀간 총 행복 차이 최소화) (-)
        return fitness

    def crossover_func(self, parents, offspring_size, _):
        """
        두개의 부모 solution을 교차하여 자녀를 생산하는 함수
        안정적 매칭(중매) 문제의 경우에는 1:1 대응되어야 하기에 서로 중복되면 안됨
        예를 들어 [1, 2, 3, 3] 은 3이 중복되어 적절한 유전자가 아님
        이 문제를 해결하기 위해 Cyclic Crossover 기법을 사용함

        Cyclic Crossover는 간단히 말해서 순열로 표현되는 염색체에 대하여 적용할 수 있는 교차 알고리즘임
        (참고: http://www.aistudy.com/biology/genetic/operator_moon.htm#_bookmark_1a54118)

        Args:
            parents: 부모 solution
            offspring_size: 자녀의 크기, 유전자 수

        return: 교차가 수행된 자녀
        """

        offspring = []  # 최종 자녀
        idx = 0  # parents 선택할 때 쓸 변수

        while len(offspring) != offspring_size[0]:  # 필요한 자녀의 크기와 현재 만들어진 자녀의 크기가 같을 때까지 반복
            # idx가 부모 parents의 원소 개수보다 크면 안되므로
            # parents.shape[0](parents의 원소 개수를 의미)로 나눈 나머지를 사용하여 계속 순환하게 함
            # 예를 들어 idx가 7이고 len(parents)가 5이면 parents[7]은 오류가 발생하므로
            # parents[7 % 5] = parents[2]로 계산하여 오류를 방지
            parent1 = parents[idx % parents.shape[0], :].copy()
            parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

            # Cyclic Crossover
            # (참고: https://codereview.stackexchange.com/a/226389)

            cycles = np.full(self.num_pairs, -1)  # 해당 위치의 사이클 번호 (-1은 아직 지정되지 않았다는 의미로 사용)
            cycle_no = 1  # 사이클 번호, 사이클이 하나 만들어질 때마다 1씩 증가시킴

            while np.any(cycles < 0):  # 아직 사이클 번호가 모두 결정될 때까지 반복
                pos = np.where(cycles < 0)[0][0]  # 결정되지 않은 지점 중 가장 왼쪽 지점의 위치

                while cycles[pos] < 0:  # pos 위치의 사이클 번호가 이미 지정되었을 때까지 (사이클이 돌아 시작한 위치로 돌아올 때까지)
                    cycles[pos] = cycle_no  # pos 위치의 사이클 번호를 현재 사이클 번호로 지정
                    pos = np.where(parent1 == parent2[pos])[0][0]  # parent2의 pos 위치에 있는 유전자 값을 parent1에서 찾아, 그 위치를 새로운 pos 값으로 지정

                cycle_no += 1  # 사이클이 완성되었으므로 사이클 번호 1 증가

            child1 = np.where(cycles % 2 == 1, parent1, parent2)  # 사이클 번호가 홀수면 parent1, 짝수면 parent2의 유전자를 사용함
            # child2 = np.where(cycles % 2 == 0, parent1, parent2)

            offspring.append(child1)  # 자녀에 추가
            # offspring.append(child2)

            idx += 1  # 새로운 parents 선택 위해 1 증가

        return np.array(offspring)

    def mutation_func(self, offspring, _):
        """
        세대에서 유전적 다양성을 유지하기 위함
        교환 돌연변이(swap mutation)를 사용함

        Args:
            offspring: 자녀

        return: 돌연변이가 수행된 자녀
        """

        for chromosome_idx in range(offspring.shape[0]):  # 각각의 자녀에 대하여
            # 임의로 두개의 점 선택
            p1 = random.randint(0, self.num_pairs - 1)
            p2 = random.randint(0, self.num_pairs - 1)

            offspring[chromosome_idx, [p1, p2]] = offspring[chromosome_idx, [p2, p1]]  # 두 점의 유전자를 서로 교환(swap)

        return offspring

    @staticmethod
    def parent_selection_func(fitness, num_parents, ga_instance):
        """
        염색체를 무작위로 추출
        steady-state 선택을 사용함

        Args:
            fitness: 적합도 평가 점수
            num_parents: 선택해야할 부모 수

        return: 선택된 부모
        """

        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])  # fitness에 따라 오름차순(작은것->큰것)으로 index 정렬
        fitness_sorted.reverse()  # 리스트 뒤집기 (내림차순 정렬, 큰것->작은것)

        parents = np.empty((num_parents, ga_instance.population.shape[1]))  # 부모 개체 저장할 리스트

        for parent_num in range(num_parents):  # 선택해야 할 부모 수만큼 반복
            parents[parent_num, :] = ga_instance.population[fitness_sorted[parent_num], :].copy()  # fitness 큰 유전자 부터 리스트에 저장

        return parents, np.array(fitness_sorted[:num_parents])

    def on_fitness(self, _, __):
        if self.idx % 100 == 0:
            print(self.idx)

        self.idx += 1

    def plot_fitness(self):
        x = np.arange(self.fitness.shape[0])
        plt.plot(x, self.fitness[:, 0], label="stable pairs")
        plt.plot(x, self.fitness[:, 1], label="hap")
        plt.plot(x, self.fitness[:, 2], label="ehap")
        plt.plot(x, self.fitness[:, 3], label="fitness")
        plt.legend()
        plt.show()

    def get_ga_instance(self):
        return pygad.GA(num_generations=self.num_generations,
                        initial_population=self.initial_population,
                        sol_per_pop=self.solution_per_population,
                        num_genes=self.num_pairs,
                        gene_type=int,
                        num_parents_mating=self.num_parents_mating,
                        crossover_probability=self.crossover_probability,
                        mutation_probability=self.mutation_probability,
                        fitness_func=self.fitness_func,
                        crossover_type=self.crossover_func,
                        mutation_type=self.mutation_func,
                        parent_selection_type=self.parent_selection_func,
                        on_fitness=self.on_fitness,
                        gene_space=list(range(self.num_pairs)))


if __name__ == '__main__':
    # 여성에 대한 남성의 선호도 목록
    # _pm = np.array([
    #     [0, 3, 2, 1],
    #     [0, 1, 2, 3],
    #     [1, 3, 2, 0],
    #     [2, 0, 1, 3]
    # ])
    #
    # # 남성에 대한 여성의 선호도 목록
    # _pw = np.array([
    #     [2, 0, 1, 3],
    #     [0, 3, 2, 1],
    #     [1, 3, 2, 0],
    #     [3, 1, 0, 2]
    # ])

    _num_pairs = 30

    _pm = np.array([sorted(np.array(range(_num_pairs)), key=lambda k: random.random()) for i in range(_num_pairs)])
    _pw = np.array([sorted(np.array(range(_num_pairs)), key=lambda k: random.random()) for i in range(_num_pairs)])

    print(_pm)
    print()
    print(_pw)

    smp = SMP(_num_pairs, _pm, _pw)

    _ga_instance = smp.get_ga_instance()
    _ga_instance.run()

    _ga_instance.plot_fitness()
    smp.plot_fitness()

    _solution, _solution_fitness, _solution_idx = _ga_instance.best_solution()
    print("Parameters of the best solution : {solution}".format(solution=_solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=_solution_fitness))
    # >> Parameters of the best solution : [0 3 1 2]
    # >> Fitness value of the best solution = 9.5
