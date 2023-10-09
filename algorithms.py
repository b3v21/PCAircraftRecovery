from copy import deepcopy
from data.build_psuedo_aus import *
import random
import numpy as np
from math import floor

NEIGHBOUR_MAPS = {}


# CODE TO GENERATE NEIGHBOUR MAP FOR EACH NODE.
def generate_neighbour_map(graph, start) -> dict:
    """
    Generates a map of all neighbours for each node in the graph.
    """

    flight_neighbours = [n for n in graph.get_neighbours(start) if n[1] is not None]
    neighbour_map = {}
    for neigh in flight_neighbours:
        if NEIGHBOUR_MAPS.get(neigh) is not None:
            neighbour_map[neigh] = NEIGHBOUR_MAPS[neigh]
            continue
        else:
            ground_neighbours = set()
            for gn in [
                n
                for n in graph.get_outgoing_nodes(neigh[0].get_name())
                if n.get_time() > neigh[0].get_time()
            ]:
                for gn2 in ground_neighbours:
                    if repr(gn) == repr(gn2):
                        break
                ground_neighbours.add(gn)
            neighbour_map[neigh] = ground_neighbours
            NEIGHBOUR_MAPS[neigh] = ground_neighbours
    return neighbour_map


def dfs_from_node(graph, start, all_paths, path=[]):
    neighbour_map = generate_neighbour_map(graph, start)

    if sum([neighbour_map[neigh] == set() for neigh in neighbour_map.keys()]):
        for neigh in neighbour_map.keys():
            new_path = deepcopy(path) + [neigh[1]]
            if new_path not in all_paths:
                all_paths.append(new_path)

    for neigh in neighbour_map.keys():
        if neighbour_map[neigh] != set():
            for gn in neighbour_map[neigh]:
                new_path = deepcopy(path) + [neigh[1]]
                if new_path not in all_paths:
                    all_paths.append(new_path)
                if len(new_path) != 3:
                    dfs_from_node(graph, gn, all_paths, new_path)
    return all_paths


def generate_all_paths(graph, all_paths=[]):
    """
    Driver function to generate all paths in the graph starting at each node in the graph.
    """

    sorted_graph = sorted(
        list(set(graph.get_all_nodes())), key=lambda x: x.get_time(), reverse=True
    )
    for start in sorted_graph:
        print(start)
        all_paths = dfs_from_node(graph, start, all_paths)
    return sorted(all_paths + [[]], key=len)

def gen_new_itins(graph, num_flights, save_name, itin_classes):
    # Number of passengers in fare class v that are originally scheduled to
    # take itinerary p

    try:
        P = generate_all_paths(graph)
    except RecursionError:
        print("ERROR: Recursion depth exceeded, please reduce itinerary length")
    print("\nitineraries created")
    
    # Limit itins used based on itin_classes
    P_copy = deepcopy(P)
    P_used = []
    itins_to_make = sum(list(itin_classes.values()))

    for _ in range(itins_to_make):
        itin = random.choice(P_copy)
        while (
            len(itin) not in list(itin_classes.keys())
            or itin_classes.get(len(itin), 0) == 0
        ):
            itin = random.choice(P_copy)
        P_copy.remove(itin)
        P_used.append(itin)
        itin_classes[len(itin)] -= 1
    
    print("\nitineraries used:")
    print(P_used, "\n")
    
    with open(f'./data/{save_name}.txt', 'w') as f:
        f.write(str(P_used))
        f.close()
        
    return P_used