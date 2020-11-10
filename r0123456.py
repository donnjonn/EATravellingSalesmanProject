import Reporter
import numpy as np
from numpy import mean
import random

# Modify the class name to match your student number.
class r0123456:

	def __init__(self):
		self.reporter = Reporter.Reporter(self.__class__.__name__)
		self.lam = 100
		self.distanceMatrix = []

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

		i = 0
		while(i < amountOfiterations):
			# calc ifitnesses
			fitnesses = []
			for ind in individuals:
				fitnesses.append(self.fitness(ind))
			print("# {}: Best fitness: {}  |  Mean fitness: {}".format(i, min(fitnesses), mean(fitnesses)))

			offspring = []
			for j in range(self.lam):
				p1 = self.selection(individuals)
				p2 = self.selection(individuals)
				offs = self.recombination(p1, p2)
				self.mutate(offs)
				offspring.append(offs)

			#self.elimination()
			individuals = offspring




			# Call the reporter with:
			#  - the mean objective function value of the population
			#  - the best objective function value of the population
			#  - a 1D numpy array in the cycle notation containing the best solution 
			#    with city numbering starting from 0
			#timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
			#if timeLeft < 0:
			#	break

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
		r1 = random.randint(0, 28)
		r2 = random.randint(0, 28)
		temp = ind.perm[r1]
		ind.perm[r1] = ind.perm[r2]
		ind.perm[r2] = temp

	def recombination(self, p1, p2):
		offspring = Individual()
		offspring.perm = [None] * amountOfVertices

		i1 = 0
		i2 = 0
		while i1 >= i2:
			i1 = random.randint(0, 28)
			i2 = random.randint(0, 28)
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
		k = 5
		selected = random.sample(individuals, k)
		fitnesses = list(map(lambda x : self.fitness(x), selected))
		index = fitnesses.index(min(fitnesses))
		return selected[index]

	def elimination(self):
		return 0

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
# Probability to mutate
alpha = 0.05

amountOfiterations = 500


# Initializations
student = r0123456()
student.optimize("tour29.csv")