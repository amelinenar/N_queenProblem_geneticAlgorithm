from collections import namedtuple
from nqueens import NQUEENS
from copy import deepcopy
import random
import numpy
import tools
import tableprint
import pandas
import click
import os

class Individual:
    """
    Class container to recorder solution and solution_value
    """
    def __init__(self,solution, fitness):
        self.solution = solution
        self.fitness = fitness

class GA:


    def __init__(self,size):
       self.problem = NQUEENS(size)


    def init_population(self,n=300):
        population = []
        for _ in range(n):
            permutation = list(range(self.problem.size))
            random.shuffle(permutation)
            population.append(Individual(solution=permutation,fitness=None))
        return population


    def evaluate(self,individual):
        """
        Return the number of conflicts on the board
        """
        return self.problem.set_nqueens(individual.solution)



    def selection(self, individuals, k, tournsize=2):
        chosen = []
        # Loop for selecting k individuals
        for _ in range(k):
            # Randomly select tournsize number of candidates from 'individuals'
            candidates = random.sample(individuals, tournsize)

            # Find the minimum fitness value among the candidates
            minimum_fitness = min(candidate.fitness for candidate in candidates)

            # Create a list of all candidates that have the minimum fitness
            candidates_with_min_fitness = [candidate for candidate in candidates if candidate.fitness == minimum_fitness]

            # If multiple candidates have the same minimum fitness, randomly select one
            if len(candidates_with_min_fitness) > 1:
                selected = random.choice(candidates_with_min_fitness)
            else:
                # If only one candidate has the minimum fitness, select that candidate
                selected = candidates_with_min_fitness[0]

            # Add the selected candidate to the list of chosen individuals
            chosen.append(selected)

        # Return the list of chosen individuals
        return chosen


    def crossover(self, ind1, ind2):

        #  Exercice 2. b)
        #
        #  Define HERE the crossover operation

        # Determine the length of the solution
        solution_length = len(ind1.solution)

        # Initialize the crossover index to the last position
        crossover_index = solution_length - 1

        # Iterate over each position in the solution
        for i in range(solution_length):
            # Check if all elements in ind1 up to position i are not in the remaining part of ind2
            if all(item not in ind2.solution[i:] for item in ind1.solution[:i + 1]):
                crossover_index = i
                break

        # Create a temporary copy of the end part of ind1's solution
        temp = ind1.solution[crossover_index:].copy()

        # Swap the end parts of the solutions of ind1 and ind2
        ind1.solution[crossover_index:] = ind2.solution[crossover_index:]
        ind2.solution[crossover_index:] = temp

        #return ind1, ind2
        return (Individual(solution=ind1.solution, fitness=ind1.fitness),
            Individual(solution=ind2.solution, fitness=ind2.fitness))
    def mutation(self,individual,indpb):
	#
	#  Exercice 2. c)
	#
	#  Define HERE the mutation operation
	#
         # Iterate over the solution using enumerate to get both index and value
        for index, value in enumerate(individual.solution):
            # Check if this element should be mutated
            if random.random() <= indpb:
                # Mutate: replace the current value with a new random value
                individual.solution[index] = random.randint(0, self.problem.size - 1)

        return individual



    def solve(self,npop,ngen, cxpb,mutpb,indpb,verbose=True):

        stats = tools.Statistics(lambda ind: ind.fitness)
        stats.register("Avg", numpy.mean)
        stats.register("Std", numpy.std)
        stats.register("Min", numpy.min)
        stats.register("Max", numpy.max)

        header = ['gen', 'nevals'] + (stats.fields if stats else [])
        all_gen = [header]

        # Generate initial population
        population = self.init_population(npop)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness is not None]
        fitnesses = map(self.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness = fit

        record = stats.compile(population) if stats else {}
        all_gen.append([0,len(invalid_ind)]+record)
        if verbose:
            tableprint.banner(f"NQUEENS -- Genetic algorithm solver -- size {self.problem.size}")
            print(tableprint.header(header,width=20))
            print(tableprint.row([0,len(invalid_ind)]+record,width=20), flush=True)

        # Begin the generational process
        for gen in range(1, ngen + 1):
            # Select the next generation individuals
            next_population = self.selection(population, len(population))

            # Apply crossover and mutation on the offspring
            for i in range(1, len(next_population), 2):
                if random.random() < cxpb:
                    next_population[i - 1], next_population[i] = self.crossover(deepcopy(next_population[i - 1]),deepcopy(next_population[i]))
                    # We need now to recompute the fitness
                    next_population[i-1].fitness = None
                    next_population[i].fitness = None

            for i in range(len(next_population)):
                if random.random() < mutpb:
                    next_population[i] = self.mutation(deepcopy(next_population[i]),indpb)
                    # We need now to recompute the fitness
                    next_population[i].fitness = None

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in next_population if not ind.fitness is not None]
            fitnesses = map(self.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness = fit


            # Replace the current population by the offspring
            population[:] = next_population

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            all_gen.append([ngen+1,len(invalid_ind)]+record)
            if verbose:
                print(tableprint.row([gen+1,len(invalid_ind)]+record,width=20), flush=True)

        return population, all_gen


def validate_proba(ctx, param, value):
    if  value >= 0 and value <= 1.0:
        return value
    else:
        raise click.BadParameter(f"Wrong {param} wrong ==> 0<= {param} <=1 ")

def check_path(ctx, param, value):
    if value is None:
        return value
    if os.path.exists(value):
        raise click.BadParameter(f"Path {param} exits already; Not overriding ")
    os.mkdir(value)
    return value


@click.command()
@click.option('--size', default=10, help='Size of the nqueens problem')
@click.option('--npop', default=100, help='Number of individual in the population')
@click.option('--ngen', default=100, help='Number of generations')
@click.option('--cxpb', default=0.5, help='Crossover probability',callback=validate_proba)
@click.option('--mutpb', default=0.2,  help='Mutation probability',callback=validate_proba)
@click.option('--indpb', default=0.2,  help='Allele mutation probability',callback=validate_proba)
@click.option('--verbose/--no-verbose')
@click.option('--save', default=None,  help='Record population and generation in a non-existing directory',callback=check_path)
def main(size,npop,ngen,cxpb,mutpb,indpb,verbose,save):
    solver_nqueens = GA(size)
    last_pop,all_gen = solver_nqueens.solve(npop,ngen, cxpb,mutpb,indpb,verbose=verbose)
    if save is not None:
        data=[]
        for ind in last_pop:
            data.append(ind.solution + [ind.fitness])
        df_pop = pandas.DataFrame(data=data, columns=[f"allele {i}" for i in range(size)]+["fitness"])
        df_pop.to_excel(os.path.join(save,"population.xlsx"))
        df_log = pandas.DataFrame(data=all_gen)
        df_log.to_excel(os.path.join(save,"log.xlsx"))





if __name__ == "__main__":
    main()
