#################
# Mellanie Martin
# Extend keras base optimizer class to implement a custom optimizer
#################

#### IMPORTS ####
from tensorflow import Sequential, Flatten, Dense
import numpy as np
import random
import os
#################

##### NOTES #####
# popSize must be a perfect square
#################

class CGA:
    def __init__(self, popSize, crossover_rate, mutation_rate, generations, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = popSize
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.dim = int(np.sqrt(self.pop_size))

    def initialize_population(self):
        '''Create initial population of individuals'''
        population = np.empty((self.dim, self.dim), dtype = object)

        for i in range(self.dim):
            for j in range(self.dim):
                population[i, j] = Sequential([
                    Flatten(input_shape=(28, 28)),
                    Dense(128, activation='relu'),
                    Dense(10, activation='softmax')
                ])

        return population

    def selection(self, individual_x, individual_y, population):
        '''Collect adjacent nodes and select parents'''
        neighbors = []

        for i in range(individual_x - 1, individual_x + 2):
            for j in range(individual_y - 1, individual_y + 2):
                neighbors.append(population[i, j])

        parents = random.sample(neighbors, 2)

        return parents

    def DX_crossover(self, parents):

    def mutation(self, offspring):

    def replace(self, individual, offspring):

    def evaluate(self, population):

    def best_individual(self, population):

    def evolve(self): # training, essentially
        population = self.initialize_population()
        # evaluate(population)
        t = self.pop_size

        while t < self.generations:
            for i in range(self.dim):
                for j in range(self.dim):
                    individual_x = i
                    individual_y = j
                    self.selection(individual_x, individual_y, population)