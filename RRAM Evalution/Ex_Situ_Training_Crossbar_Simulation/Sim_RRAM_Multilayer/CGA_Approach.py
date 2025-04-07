#################
# Mellanie Martin
# Implement Cellular Genetic Algorithm, an evolutionary based training algorithm
#################

#### IMPORTS ####
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
import numpy as np
import random
import math
import os
#################

class CGA:
    def __init__(self, popSize, crossover_rate, mutation_rate, generations, x_val, y_val, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = popSize
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.x_val = x_val
        self.y_val = y_val

    def initialize_population(self):
        '''Create initial population of individuals'''
        population = {} # Potential fix; alter population to resemble a toroidal mesh

        for i in range(self.pop_size):

            key = i # index

            model = Sequential([
                Flatten(input_shape=(28, 28)),
                Dense(128, activation='relu'),
                Dense(10, activation='softmax')
            ])
            model.compile(optimizer = SGD(learning_rate = 0.0), loss = 'categorical_crossentropy', metrics = ['accuracy'])

            inital_weights = []
            for layer in model.layers:
                inital_weights.extend(layer.get_weights())

            # evaluate(population) in algorithm
            fitness = self.get_fitness(model)

            #                   [0]             [1]                     [2]
            population[key] = {'model': model, 'wb': inital_weights, 'fitness': fitness}

        return population

    def selection(self, individual, population):
        '''Collect adjacent nodes and select parents'''
        neighbors = []

        for i in range(individual - 2, individual + 3):
            index = i % self.pop_size # Wrap around
            neighbors.append(population[index])

        parents = random.sample(neighbors, 2)

        return parents

    def DX_crossover(self, parents, best_individual, t):
        '''Perform Damped Crossover (DX) to create offspring'''
        mother = self.flatten_weights(parents[0][1])
        father = self.flatten_weights(parents[1][1])
        best = self.flatten_weights(best_individual[1])

        avg_gene = [] # average of each gene
        diff_gene = [] # how far best individual is from average for each gene - 'steers'
        for i, j, k in zip(mother, father, best):
            avg = (i + j)/2
            diff = k - avg
            avg_gene.append((i + j)/2)
            diff_gene.append(diff)

        # Compute damped oscillation factor
        # ratio = A * e^(-Ct) * sin(P + 99.75 * t)
        # A = 1 (initial amplitude), C = 3 (how fast damping happens), P = .5 (phase shift), t = # of evalutions
        ratio = 1.0 * (math.exp(-1 * 3 * t)) * math.sin(.5 + 99.75 * t)

        inc_gene = [] # how much to move towards the best individual
        for i in diff_gene:
            inc_gene.append(i * (1 + ratio))

        # finally create offspring
        o1 = []
        o2 = []
        for i, j, k in zip(mother, father, inc_gene):
            o1.append(i + k) # o1_i = p1_i + inc_i
            o2.append(j + k) # o2_i = p2_i + inc_i

        return o1, o2

    def uniform_mutation(self, offspring): # Potential fix; use non_uniform, or another mutation algorithm
        '''Perform mutation on offspring'''
        for child in offspring:
            for gene in child:
                r = random.uniform(0, 1)
                gene = gene + (r - .5) * .5 # may need to be altered to ensure it is in bounds

        return offspring
                

    def replace(self, individual, offspring):
        '''Replace individual with offspring if either offspring is fitter'''
        for child in offspring:
            unflattened_child = self.unflatten_weights(child)
            model = Sequential([
                Flatten(input_shape=(28, 28)),
                Dense(128, activation='relu'),
                Dense(10, activation='softmax')
            ])
            model.set_weights(unflattened_child)
            model.compile(optimizer = SGD(learning_rate = 0.0), loss = 'categorical_crossentropy', metrics = ['accuracy'])

            fitness = self.get_fitness(model)
            if fitness > individual['fitness']:
                child_dict = {'model': model, 'wb': child, 'fitness': fitness}
                individual.update(child_dict)

    def get_fitness(self, model):
        '''Evaluate solution fitness with just a forward pass'''
        loss, acc = model.evaluate(self.x_val, self.y_val, verbose = 0)
        return acc

    def best_individual(self, population):
        '''Get current best individual in population.
        Returns best_individual as its key in the population'''
        best_individual = -1
        best_fitness = -1

        for key in population:
            fitness = population[key]['fitness']
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = key

        return best_individual

    def evolve(self):
        '''Putting it all together - What will be called from Perceptron_Tensorflow_Weights.py'''
        population = self.initialize_population()
        t = self.pop_size
        num_generations = 1

        while t < self.generations:
            for key in population:
                parents = self.selection(key, population)
                best = self.best_individual(population)
                offspring = self.DX_crossover(parents, population[best], num_generations)
                offspring = self.uniform_mutation(offspring)
                self.replace(population[key], offspring)
                num_generations += 2

        best = self.best_individual(population)
        best_individual = population[best]
        return best_individual

    def flatten_weights(self, wb):
        '''Flatten weights + biases of every layer into a genome'''
        return np.concatenate([w.flatten() for w in wb])
    
    def unflatten_weights(self, wb_vector):
        '''Unflatten weights + biases so they can be applied to network'''
        shapes = [ # Hardcoded for MNIST dataset
            (784, 128),
            (128,),
            (128, 10),
            (10,)
        ]

        reshaped = []
        i = 0
        for shape in shapes:
            size = np.prod(shape)
            reshaped.append(wb_vector[i:i + size].reshape(shape))
            i += size
        return reshaped