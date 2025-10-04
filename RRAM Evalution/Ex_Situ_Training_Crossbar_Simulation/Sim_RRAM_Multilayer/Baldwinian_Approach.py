#################
# Mellanie Martin
# Implement a Baldwinian algorithm, a hybrid between CGA and gradient-based learning
#################

#### IMPORTS ####
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
import numpy as np
import random
import math
import os

import Mutation_Algos
#################

class Baldwinian:
    def __init__(self, popSize, crossover_rate, mutation_rate, generations, x_train, x_test, y_train, y_test, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = popSize
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

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
            model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

            initial_weights = []
            for layer in model.layers:
                initial_weights.extend(layer.get_weights())

            # Get upper + lower bound of initialized weights
            #weights, biases = model.layers[1].get_weights()

            # evaluate(population) in algorithm
            fitness = self.get_fitness(model)

            #                   [0]             [1]                     [2]                 [3]                 [4]
            population[key] = {'model': model, 'wb': initial_weights, 'fitness': fitness} #'lb': weights.min(), 'ub': weights.max()}

        return population

    def selection(self, individual, population):
        '''Collect adjacent nodes and select parents'''
        neighbors = []

        for i in range(individual - 2, individual + 3):
            index = i % self.pop_size # Wrap around
            neighbors.append(population[index])

        parents = random.sample(neighbors, 2)

        return parents

    def DX_crossover(self, parents, best_individual, t, rate, rank, pop_size):
        '''Perform Damped Crossover (DX) to create offspring'''
        p_m = rate * (1 - ((rank - 1)/(pop_size - 1)))

        if random.random() > p_m: # No crossover in reproduction
            o1 = self.flatten_weights(parents[0][1])
            o2 = self.flatten_weights(parents[1][1])
            return o1, o2
        
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
            model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

            weights, biases = model.layers[1].get_weights()

            fitness = self.get_fitness(model)
            if fitness > individual['fitness']:
                child_dict = {'model': model, 'wb': child, 'fitness': fitness} #'lb': weights.min(), 'ub': weights.max()}
                individual.update(child_dict)

    def get_fitness(self, model):
        '''Evaluate solution fitness with just a forward pass'''
        model.fit(self.x_train, self.y_train, epochs = 1, verbose = 0) # Baldwinian approach
        loss, acc = model.evaluate(self.x_test, self.y_test, verbose = 0)
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
                rankings = sorted(population.items(), key = lambda item: item[1]['fitness'], reverse = True)
                chromosomal_rank = rankings.index(key) + 1
                parents = self.selection(key, population)
                best = self.best_individual(population)
                offspring = self.DX_crossover(parents, population[best], num_generations, self.crossover_rate)
                offspring = Mutation_Algos.adaptive_uniform(offspring, self.mutation_rate, chromosomal_rank, self.pop_size)
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