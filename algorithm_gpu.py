import random
import numpy as np
from deap import base, creator, tools, algorithms
from Agents.RandomAgent import RandomAgent as ra
from Agents.AdrianHerasAgent import AdrianHerasAgent as aha
from Agents.AlexPastorAgent import AlexPastorAgent as apa
from Agents.AlexPelochoJaimeAgent import AlexPelochoJaimeAgent as apja
from Agents.CarlesZaidaAgent import CarlesZaidaAgent as cza
from Agents.CrabisaAgent import CrabisaAgent as ca
from Agents.EdoAgent import EdoAgent as ea
from Agents.PabloAleixAlexAgent import PabloAleixAlexAgent as paaa
from Agents.SigmaAgent import SigmaAgent as sa
from Agents.TristanAgent import TristanAgent as ta
from Managers.GameDirector import GameDirector
# from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

AGENTS = [ra, aha, apa, apja, cza, ca, ea, paaa, sa, ta]
N = 100
fitness_cache = {}

def swap_probabilities_np(prob_array):
    sorted_indices = np.argsort(prob_array) 
    swapped_probs = prob_array.copy() 
    n = len(prob_array)
    
    for i in range(n // 2):
        low_idx, high_idx = sorted_indices[i], sorted_indices[n - i - 1]
        swapped_probs[low_idx], swapped_probs[high_idx] = swapped_probs[high_idx], swapped_probs[low_idx]

    return swapped_probs / np.sum(swapped_probs)


def evaluate(individual):

    global fitness_cache
    individual_tuple = tuple(individual)

    if individual_tuple in fitness_cache:
        return fitness_cache[individual_tuple]
    
    total_wins = 0
    player = random.choices(AGENTS, weights=individual, k=1)[0]
    aux = np.delete(individual, AGENTS.index(player))
    aux_swap = swap_probabilities_np(aux)

    for _ in range(N):
        opponents = np.random.choice([agent for agent in AGENTS if agent != player], size=3, p=aux_swap)
        players = [player] + list(opponents)
        random.shuffle(players)

        try:
            game_director = GameDirector(agents=players, max_rounds=200, store_trace=False)
            game_trace = game_director.game_start(print_outcome=False)

            last_round = max(game_trace["game"].keys(), key=lambda r: int(r.split("_")[-1]))
            last_turn = max(game_trace["game"][last_round].keys(), key=lambda t: int(t.split("_")[-1].lstrip("P")))
            victory_points = game_trace["game"][last_round][last_turn]["end_turn"]["victory_points"]

            sorted_victory_points = dict(sorted(victory_points.items(), key=lambda item: int(item[1]), reverse=True))
            players_ranking = list(sorted_victory_points.keys())
            ind_player = players.index(player)
            pos = int(players_ranking[ind_player].lstrip("J"))
            if pos < 3:
                total_wins += (1 / 2**pos)

        except Exception as e:
            print(f"Error in game simulation: {e}")

    fitness = total_wins / N
    fitness_cache[individual_tuple] = (fitness,)

    return (fitness,)


class CatanGA:
    def __init__(self, 
                 population_size=20, 
                 generations=50, 
                 mutation_prob=0.2, 
                 crossover_prob=0.5, 
                 selection_method="tournament", 
                 tournament_size=3,
                 mutation_sigma=0.1, 
                 mutation_indpb=0.2, 
                 num_games_per_individual=5):

        self.population_size = population_size
        self.generations = generations
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.mutation_sigma = mutation_sigma
        self.mutation_indpb = mutation_indpb
        self.num_games_per_individual = num_games_per_individual
        self.toolbox = base.Toolbox()
        self.logbook = tools.Logbook() 
        self.setup_deap()

    def setup_deap(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # def generate_probabilities():
        #     probs = np.random.dirichlet(np.ones(len(AGENTS)), size=1)[0]
        #     probs = probs.tolist()
        #     self.normalize(probs)
        #     return probs

        def generate_probabilities():
            probs = np.random.dirichlet(np.ones(len(AGENTS)), size=1)[0]
            return np.round(probs, 4)

        self.toolbox.register("individual", tools.initIterate, creator.Individual, generate_probabilities)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        if self.selection_method == "tournament":
            self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        elif self.selection_method == "roulette":
            self.toolbox.register("select", tools.selRoulette)
        elif self.selection_method == "best":
            self.toolbox.register("select", tools.selBest)
        else:
            raise ValueError("Invalid selection method. Choose 'tournament', 'roulette', or 'best'.")

        self.toolbox.register("mate", self.cx_blend_weighted)
        # self.toolbox.register("mate", self.cx_two_point_normalized)
        self.toolbox.register("mutate", self.mut_gaussian_normalized, sigma=self.mutation_sigma, indpb=self.mutation_indpb)

        self.toolbox.register("evaluate", evaluate)

        # num_cores = multiprocessing.npu_count()  # Get available npU cores
        # self.pool = multiprocessing.Pool(processes=num_cores)  # Create a process pool
        self.pool = ThreadPoolExecutor()
        self.toolbox.register("map", self.pool.map)

    def normalize(self, individual):
        total = sum(individual)
        for i in range(len(individual)):
            individual[i] = np.round(individual[i] / total, 4)

        difference = 1.0 - sum(individual)
        if difference != 0:
            max_index = individual.index(max(individual))
            individual[max_index] = np.round(individual[max_index] + difference, 4)
        return

    def cx_blend_weighted(self, ind1, ind2, alpha=0.5):

        for i in range(len(ind1)):
            mix = (1 - alpha) * ind1[i] + alpha * ind2[i]
            ind1[i] = mix
            ind2[i] = mix

        # Normalize both offspring
        self.normalize(ind1)
        self.normalize(ind2)

        return ind1, ind2


    def cx_two_point_normalized(self, ind1, ind2):
        tools.cxTwoPoint(ind1, ind2) 

        self.normalize(ind1)
        self.normalize(ind2)

        return ind1, ind2

    def mut_gaussian_normalized(self, individual, sigma, indpb):
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] += random.gauss(0, sigma)
                individual[i] = max(individual[i], 0)
                individual[i] = min(individual[i], 1)
        self.normalize(individual)
        return individual,

    def run(self):
        global fitness_cache
        population = self.toolbox.population(n=self.population_size)
        hall_of_fame = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)

        self.logbook.clear()
        fitness_cache = {}

        for gen in range(self.generations):
            fitnesses = list(self.toolbox.map(evaluate, population))

            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            
            record = stats.compile(population) 
            self.logbook.record(gen = gen, **record) 

            best_ind = tools.selBest(population, 1)[0]
            print(f"Generation {gen} - Best Individual: {[float(x) for x in best_ind]}, Max Fitness: {record['max']:.4f}, Avg Fitness: {record['avg']:.4f}")

            hall_of_fame.update(population)

            population = self.toolbox.select(population, k=len(population)//2)
            population.extend(algorithms.varAnd(population, self.toolbox, cxpb=self.crossover_prob, mutpb=self.mutation_prob))

        # self.pool.close()
        # self.pool.join()
        self.pool.shutdown()
        agent_ind = hall_of_fame[0].index(max(hall_of_fame[0]))
        return hall_of_fame[0], AGENTS[agent_ind]
