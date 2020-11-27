import Reporter
import numpy as np
from numpy import mean
import random

random.seed(30)
np.random.seed(30)

# Modify the class name to match your student number.
class r0123456:

    def __init__(self, params):
        self.reporter = Reporter.Reporter("grid_search_results/" + file + "/" + self.__class__.__name__ + 'iter=' + str(amountOfiterations) + '_stopcrit' + str(stopIteratingAfter) + '_lambda=' + str(params["lam"]) + '_alpha=' + str(params["alpha"]) + '_k=' + str(params["k"]) + '_' + file)
        self.lam = params["lam"]
        self.alpha = params["alpha"]
        self.k = params["k"]
        self.distanceMatrix = []
        self.listMeans = np.zeros(shape=(amountOfiterations))
        self.stopIter = 0

    # The evolutionary algorithm's main loop
    def optimize(self, filename, optimum=None):
        # Read distance matrix from file.		
        file = open(filename)
        self.distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        # Your code here.
        individuals = self.initialize()

        #testind1 = individuals[0]
        #testind2 = individuals[1]

        #self.pathLegit(testind1)
        #print("parent1: ", testind1.perm)
        #print("parent2: ", testind2.perm)
        #self.recombination(testind1, testind2)
        bestOverall = 0
        #bestObjective = 0
        #meanObjective = 0
        i = 0

        # "or" part explanation: 
        #       True IF optimum_reference is not given (as an argument of optimize) 
        #            ELSE (if optimum_reference given), we compare this value with the current meanObjective (and return True if meanObjective <= optimum_reference)
        while( (i < amountOfiterations and self.stopIter <= stopIteratingAfter) ):
            index = 0
            # calc fitnesses
            fitnesses = np.zeros(shape=(self.lam))
            for k in range(len(individuals)):
                fitnesses[k] = self.fitness(individuals[k])
            bestObjective = np.amin(fitnesses)
            index = np.argmin(fitnesses)
            bestSolution = np.array(individuals[index].perm)
            meanObjective = np.mean(fitnesses)
            self.listMeans[i] = meanObjective
            if bestObjective < bestOverall or i == 0:
                bestOverall = bestObjective
            if (i != 0 and round(self.listMeans[i], 0) == round(self.listMeans[i-1], 0)):
                self.stopIter += 1
            else:
                self.stopIter = 0
            offspring = [None] * self.lam
            for j in range(self.lam):
                p1 = self.selection(individuals)
                p2 = self.selection(individuals)
                offs = self.recombination(p1, p2)
                if random.uniform(0, 1) < self.alpha:
                    self.mutate(offs)
                offspring[j] = offs
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
            print("{}/{}: # {}: Best fitness: {:.10f}  |  Mean fitness: {:.10f} | Best fitness overall: {:.10f} | Time left: {}".format(exp_counter, total_experiments, i, bestObjective, meanObjective, bestOverall, timeLeft))
            
            if self.stopIter == stopIteratingAfter:
                print("Stopped iterating after {} of the same mean fitness value".format(stopIteratingAfter))

            if timeLeft < 0:
                break

            if i == amountOfiterations:
                print("Reached maximum amount of iterations ({})".format(amountOfiterations))

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
        selected = random.sample(individuals, self.k)
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


    # return True if path has no duplicates and contains all possible nodes
    # return False otherwise
    def pathLegit(self, path):

        nb_nodes = len(self.distanceMatrix[0]) 
        nodes = [vert for vert in range(nb_nodes)] # ordered list of all possible nodes
        # np.unique(path.perm) takes the path, removes deplicates and order the list

        # compare 2 ordered list (must be exactly the same)
        if(np.array_equal(np.array(nodes),np.unique(path.perm))):
            return True
        else:
            return False


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
#lam = 100
# Probability to mutate
#alpha = 0.2
#k = 5





#gs_alpha = [0.01, 0.05, 0.075, 0.1, 0.25, 0.5]
#gs_k = [1, 2, 3, 4, 5, 6, 10]


#test_experiments = [{"lam":100, "alpha":0.01, "k": 1},{"lam":100, "alpha":0.01, "k": 2}]

optimums = { "tour29": 27200, "tour100": 7350, "tour194": 9000, "tour929": 95300}
amountOfiterations = 4000
stopIteratingAfter = 500
k_elimination = 5

# 100 tour29 experiments
#gs_lam = [100, 200, 300, 500]
#gs_alpha = [0.1, 0.2, 0.3, 0.4]
#gs_k = [1, 2, 3, 4, 5]

# 0.04 graph
gs_lam = [50, 75, 100, 200, 300, 500, 600]
gs_alpha = [0.04]
gs_k = [4]

experiments = []
for l in gs_lam:
    for a in gs_alpha:
        for k in gs_k:
            experiments.append({"lam":l, "alpha":a, "k":k})

#gs_alpha = [0.2, 0.3]
#gs_k = [5]

#for l in gs_lam:
#    for a in gs_alpha:
#        for k in gs_k:
#            experiments.append({"lam":l, "alpha":a, "k":k})


#experiment counter for logging purposes
total_experiments = len(experiments)
exp_counter = 1

#params = {"lam":100, "alpha":0.05, "k": 4}
# Initializations
file = 'tour' + str(amountOfVertices)
#student = r0123456(params)
#student.optimize(file + '.csv')

for e in experiments:
    student = r0123456(e)
    student.optimize(file + '.csv', optimums["tour29"])

    exp_counter += 1

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