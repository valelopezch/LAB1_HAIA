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
import random

AGENTS = [ra, aha, apa, apja, cza, ca, ea, paaa, sa, ta]
players = random.sample(AGENTS, 4)
player = random.sample(players, 1)[0]


print(f'Players: {players}')
print(f'Players: {len(set(players))}')
print(f'Player: {player}')
total_wins = 0
try:
    game_director = GameDirector(agents = players, max_rounds = 200, store_trace = False)
    game_trace = game_director.game_start(print_outcome = False)
except Exception as e:
    print(f"Error: {e}")

last_round = max(game_trace["game"].keys(), key=lambda r: int(r.split("_")[-1]))
last_turn = max(game_trace["game"][last_round].keys(), key=lambda t: int(t.split("_")[-1].lstrip("P")))
victory_points = game_trace["game"][last_round][last_turn]["end_turn"]["victory_points"]
sorted_victory_points = dict(sorted(victory_points.items(), key=lambda item: int(item[1]), reverse=True))

players_ranking = list(sorted_victory_points.keys())
ind_player = players.index(player)
pos = int(players_ranking[ind_player].lstrip("J"))
# winner = max(victory_points, key=lambda p: int(victory_points[p]))
# if players.index(player) == int(winner.lstrip("J")):
total_wins += (1/2**pos)

winner = max(victory_points, key=lambda player: int(victory_points[player]))
print(sorted_victory_points)
print(players_ranking)
print(f'Winner: {winner}')
print(total_wins)


