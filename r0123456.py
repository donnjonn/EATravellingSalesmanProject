import Reporter
import numpy as np
from numpy import mean
import random

random.seed(30)
np.random.seed(30)

# Modify the class name to match your student number.
class r0123456:

    def __init__(self):
        self.reporter = Reporter.Reporter(self.__class__.__name__ + 'iter=' + str(amountOfiterations) + '_stopcrit' + str(stopIteratingAfter) + '_lambda=' + str(lam) + '_alpha=' + str(alpha) + '_k=' + str(k) + '_' + file)
        self.lam = lam
        self.distanceMatrix = []
        self.listMeans = []
        self.stopIter = 0

    # The evolutionary algorithm's main loop
    def optimize(self, filename):
        # Read distance matrix from file.		
        file = open(filename)
        self.distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here.
        individuals = self.initialize()

        testind1 = individuals[0]
        testind2 = individuals[1]
        print("parent1: ", testind1.perm)
        print("parent2: ", testind2.perm)
        self.recombination(testind1, testind2)
        bestOverall = 0
        bestObjective = 0
        meanObjective = 0
        i = 0
        while(i < amountOfiterations and self.stopIter <= stopIteratingAfter):
            index = 0
            # calc ifitnesses
            fitnesses = []
            for ind in individuals:
                fitnesses.append(self.fitness(ind))
            bestObjective = min(fitnesses)
            index = fitnesses.index(bestObjective)
            bestSolution = np.array(individuals[index].perm)
            meanObjective = np.mean(fitnesses)
            self.listMeans.append(meanObjective)
            if bestObjective < bestOverall or i == 0:
                bestOverall = bestObjective
            #print(round(self.listMeans[-1], 0), self.stopIter)
            if (i != 0 and round(self.listMeans[-1], 0) == round(self.listMeans[-2], 0)):
                self.stopIter += 1
            else:
                self.stopIter = 0
            offspring = []
            for j in range(self.lam):
                p1 = self.selection(individuals)
                p2 = self.selection(individuals)
                offs = self.recombination(p1, p2)
                if random.uniform(0, 1) < alpha:
                    self.mutate(offs)
                offspring.append(offs)
            newpop = []
            for j in range(self.lam):
                #newpop.append(self.elimination_select_half_best(individuals, offspring, self.distanceMatrix))
                newpop.append(self.elimination(individuals, offspring))
            #print(len(newpop))
            individuals = newpop
            # Call the reporter with:
            #  - the mean objective function value of the population
            #  - the best objective function value of the population
            #  - a 1D numpy array in the cycle notation containing the best solution 
            #    with city numbering starting from 0
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            print("# {}: Best fitness: {:.10f}  |  Mean fitness: {:.10f} | Best fitness overall: {:.10f} | Time left: {}".format(i, bestObjective, meanObjective, bestOverall, timeLeft))
            
            if self.stopIter == stopIteratingAfter:
                print("Stopped iterating after {} of the same mean fitness value".format(stopIteratingAfter))

            if timeLeft < 0:
                break

            i += 1
        # Your code here.

    def fitness(self, ind):
        sum = 0
        for i in range(len(ind.perm)-1):
            sum += self.distanceMatrix[ind.perm[i]][ind.perm[i+1]]
        return sum

    def initialize(self):
        individuals =  list(map(lambda x : Individual(), [1]*self.lam))
        for ind in individuals:
            if not self.pathExists(ind):
                # reinitialize to legit paths
                continue
            else:
                continue

        return individuals

    def mutate(self, ind):
        r1 = random.randint(0, amountOfVertices-1)
        r2 = random.randint(0, amountOfVertices-1)
        temp = ind.perm[r1]
        ind.perm[r1] = ind.perm[r2]
        ind.perm[r2] = temp

    def recombination(self, p1, p2):
        offspring = Individual()
        offspring.perm = [None] * amountOfVertices

        i1 = 0
        i2 = 0
        while i1 >= i2:
            i1 = random.randint(0, amountOfVertices-1)
            i2 = random.randint(0, amountOfVertices-1)
        offspring.perm[i1:i2] = p1.perm[i1:i2]

        for i in range(len(p2.perm[i1:i2])):
            current_index = i1+i
            v = p2.perm[current_index]
            if v in offspring.perm:
                continue

            while offspring.perm[current_index] is not None:
                v2 = p1.perm[current_index]
                new_index = p2.perm.index(v2)
                current_index = new_index


            offspring.perm[current_index] = v
        for i in range(len(offspring.perm)):
            if offspring.perm[i] is None:
                offspring.perm[i] = p2.perm[i]
        return offspring

    def selection(self, individuals):
        # k-tournament selection
        selected = random.sample(individuals, k)
        fitnesses = list(map(lambda x : self.fitness(x), selected))
        index = fitnesses.index(min(fitnesses))
        return selected[index]

    def elimination(self, parents, offspring):
        totalpop = parents + offspring
        selected = random.sample(totalpop, k_elimination)
        fitnesses = list(map(lambda x : self.fitness(x), selected))
        index = fitnesses.index(min(fitnesses))
        return selected[index]


    def elimination_select_half_best(self, parents, offsprings, distanceMatrix):

        totalPop = parents + offsprings

        dtype = [('index', int), ('length', float)]

        fitnesses = np.array([ (i , self.length(ind, distanceMatrix)) for i, ind in enumerate(totalPop) ], dtype=dtype)

        sortedd = np.sort(fitnesses, order='length')
        

        retour = [totalPop[ele[0]] for i, ele in enumerate(sortedd) if i<len(parents)]

        return retour




    # Helperfunctions
    def pathExists(self, ind):
        # To check if path is legit (later)
        return True

        return 0

class Individual:

    def __init__(self):
        self.perm = list(np.random.permutation(amountOfVertices))


# Executed code starts here
# Parameters
amountOfVertices = 29
lam = 500
# Probability to mutate
alpha = 0.2

amountOfiterations = 3000

k = 5
k_elimination = 5

stopIteratingAfter = 500



# Initializations
file = 'tour29'
student = r0123456()
student.optimize(file + '.csv')

experiments = []

# Parameter suggestions:
#   PMX length of subset of one of the parents

# Experiments=
# lam   = 50 100 200 300 400 500 600 1000 (500 best)
# alpha = 0.01 0.05 0.1 0.5 (0.05)
# k     = 5 10 20
#

# with lam 500:
# alpha 0.01 0.025 0.05 0.075 0.1 (0.1 best)
# k 5 10