import random

'''
Uniform Mutation; causes a LOT of variation in results
'''
def uniform_mutation(offspring, rate): 
    '''Perform mutation on offspring'''
    
    for child in offspring:
        for gene in child:
            if random.random() > rate: # no mutation in reproduction
                continue
            r = random.uniform(0, 1)
            gene = gene + (r - .5) * .5 # may need to be altered to ensure it is in bounds

    return offspring

def non_uniform_mutation(offspring, rate, bounds, current_eval, max_evals):
    for child in offspring:
        for gene in child:
            r = random.random()
            if r < rate: 
                gene = gene + change_over_td(current_eval, max_evals, r, bounds[1] - gene)
            else: gene = gene + change_over_td(current_eval, max_evals, r, gene - bounds[0])

    return offspring

def change_over_td(t, T, r, d, b = .5):
    exp = (1 - t/T) ** b
    return d * (1 - (r ** exp))