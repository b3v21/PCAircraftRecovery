from copy import deepcopy
from data.build_psuedo_aus import *
import random
import numpy as np
from math import floor


# CODE TO GENERATE NEIGHBOUR MAP FOR EACH NODE.
def generate_neighbour_map(graph, start) -> dict:
    """
    Generates a map of all neighbours for each node in the graph.
    """
    flight_neighbours = [n for n in graph.get_neighbours(start) if n[1] is not None]
    neighbour_map = {}
    for neigh in flight_neighbours:
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
                dfs_from_node(graph, gn, all_paths, new_path)
    return all_paths


def generate_all_paths(graph, all_paths=[]):
    """
    Driver function to generate all paths in the graph starting at each node in the graph.
    """

    for start in graph.adj_list.keys():
        all_paths = dfs_from_node(graph, start, all_paths)
    return sorted(all_paths + [[]], key=len)
