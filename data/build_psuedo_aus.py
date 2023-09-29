#########################################################################################
# INITIAL AIRPORTS:
# SYD - Sydney Airport (HUB) (-33.94999577372875, 151.18170512494225)
# MEL - Melbourne Airport (-37.673333, 144.843333)
# BNE - Brisbane Airport (-27.386944, 153.1175)
# PER - Perth Airport (-31.940278, 115.966944)
# ADL - Adelaide Airport (-34.945, 138.530556)
# OOL - Gold Coast Airport (-28.164444, 153.505278)
# CNS - Cairns Airport (-16.885833, 145.755278)
# HBA - Hobart International Airport (-42.836111, 147.510278)
# CBR - Canberra Airport (-35.306944, 149.195)
# TSV - Townsville Airport (-19.2525, 146.765278)

# AIRLINE: RANTAS (Rantas Airlines) (Not associated or inspired by QANTAS in any way)
# NOTES:
#   * ALL MAINTENANCE IS DONE IN SYD
#   * IS IT ASSUMED ALL PLANES FLY TO EACHOTHER BUT A lARGE
#     PORTION OF FLIGHTS FLY ARE O.T.F: X --> SYD and SYD --> X
#   * AIM FOR ~1000 FLIGHTS OVER A 3 DAY PERIOD
#   * FLIGHT TIME WILL BE GENERATED FROM A NORMAL DISTRIBUTION
#     WITH A MEAN & STANDARD DEVIATION BASED OFF THE DISTANCE OF NODES
#########################################################################################

import copy
import random
import numpy as np
from math import floor, ceil
import geopy.distance

from random import randrange

################################# HELPER FUNCTIONS #################################


def divide_number(number, divider, min_value, max_value):
    """
    Code from online, used to divide a number into a given number of parts
    """

    result = []
    for i in range(divider - 1, -1, -1):
        part = randrange(min_value, min(max_value, number - i * min_value + 1))
        result.append(part)
        number -= part
    return result


def calculate_time(loc1: tuple[float, float], loc2: tuple[float, float]) -> float:
    """
    Calculates the time it takes to fly between two locations
    """

    # Time is distance in km / average speed of 700km/h + 30min for takeoff/landing
    return round(geopy.distance.geodesic(loc1, loc2).km / 700 + 0.5, 1)


TIME_HORIZON = 72
AIRPORTS = ["SYD", "MEL", "BNE", "PER", "ADL", "OOL", "CNS", "HBA", "CBR", "TSV"]
COORDS = [
    (-33.949995, 151.181705),
    (-37.673333, 144.843333),
    (-27.386944, 153.1175),
    (-31.940278, 115.966944),
    (-34.945, 138.530556),
    (-28.164444, 153.505278),
    (-16.885833, 145.755278),
    (-42.836111, 147.510278),
    (-35.306944, 149.195),
    (-19.2525, 146.765278),
]
WEIGHTS = [0.25, 0.15, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]


class Node:
    def __init__(self, node_id, name, lat, long, time):
        self.id = node_id
        self.name = name
        self.lat = lat
        self.long = long
        self.time = time

    def __repr__(self):
        return f"{self.name}: {self.time}"

    def __str__(self):
        return f"{self.name}: {self.time}"

    def __eq__(self, __value: object) -> bool:
        if self.name == __value.name and self.time == __value.time:
            return True

    def __hash__(self) -> int:
        return hash(str(self))

    def get_id(self):
        return self.id

    def get_name(self):
        return self.name

    def new_time_copy(self, time, node_id):
        new_node = copy.deepcopy(self)
        new_node.time = time
        new_node.id = node_id
        return new_node

    def get_time(self):
        return self.time


class AdjanecyList:
    def __init__(self):
        self.adj_list = {}

    def add_node(self, node: Node, neighbours: list[Node], flight_id: int):
        self.adj_list[node] = []
        for neighbour in neighbours:
            self.adj_list[node].append((neighbour, flight_id))

    def get_nodes(self, name):
        result = []
        for node in self.adj_list:
            if node.name == name:
                result.append(node)
        return result

    def add_neigh_to_node(self, node: Node, neighbour: Node, flight_id: int):
        if node in self.adj_list:
            self.adj_list[node].append((neighbour, flight_id))
        else:
            self.adj_list[node] = [(neighbour, flight_id)]

    def get_neighbours(self, node: Node) -> list[Node]:
        return self.adj_list.get(node, [])

    def get_all_nodes(self) -> list[Node]:
        return list(self.adj_list.keys()) + sum(
            [[t[0] for t in L] for L in self.adj_list.values()], []
        )

    def count_node_locations(self):
        counts = {k: 0 for k in AIRPORTS}
        for node in self.adj_list.keys():
            if True in [(lambda x: x[1] != None)(n) for n in self.get_neighbours(node)]:
                counts[node.name] += 1
        return counts

    def count_all_flights(self):
        count = 0
        for node in self.adj_list.keys():
            count += sum(
                [(lambda x: x[1] != None)(n) for n in self.get_neighbours(node)]
            )
        return count

    def get_first_n_flights(self, n):
        pass

    def __repr__(self):
        output = ""
        for node, neighbours in self.adj_list.items():
            output += "{}: {}\n".format(node, str(neighbours))
        return output


def generate_flight_arc(
    graph: AdjanecyList,
    node: Node,
    default_nodes: list[Node],
    current_flight_id: int,
    current_node_id: int,
) -> None:
    """
    Generate arc which travel to a randomized airport at a randomized flight time.
    """

    # Randomise destination airport
    dest_node = random.choices(default_nodes, weights=WEIGHTS)[0]
    while dest_node == node:
        dest_node = random.choices(default_nodes, weights=WEIGHTS)[0]

    flight_time = calculate_time((node.lat, node.long), (dest_node.lat, dest_node.long))

    # Randomise time flight is scheduled to depart
    departure_time = round(
        random.random() * TIME_HORIZON - ceil(flight_time), 1
    )  # TODO: this needs to be changed when flight times are changed
    departure_node = node.new_time_copy(departure_time, current_node_id)

    # if departure_node in graph.adj_list:
    graph.add_neigh_to_node(
        departure_node,
        dest_node.new_time_copy(departure_time + flight_time, current_node_id + 1),
        current_flight_id,
    )


def generate_ground_arcs(graph: AdjanecyList) -> None:
    """
    Given the current flight arcs in graph, generate all possible ground arcs from a given node,
    i.e arcs which stay at the same airport until the next available flight.

    To generate all ground arcs for a particular airport, we must first find all nodes with the
    same airport name which are found in the . Then we must order those nodes by arrival time
    """

    flight_nodes = graph.get_all_nodes()

    for n in flight_nodes:
        for nd in flight_nodes:
            if n.get_name() == nd.get_name() and n.get_time() < nd.get_time():
                graph.add_neigh_to_node(n, nd, None)


def create_graph(flight_distribution):
    # Create default nodes
    default_nodes = [
        Node(0, airport, coords[0], coords[1], None)
        for airport, coords in zip(AIRPORTS, COORDS)
    ]

    current_node_id = 0
    current_flight_id = 0
    graph = AdjanecyList()

    for count, node in enumerate(default_nodes):
        flights_for_node = flight_distribution[count]

        for _ in range(flights_for_node):
            generate_flight_arc(
                graph,
                node,
                default_nodes,
                current_node_id,
                current_flight_id,
            )
            current_node_id += 1
            current_flight_id += 1

    generate_ground_arcs(graph)

    return graph


def itinerary_builder(
    graph: AdjanecyList, length: int, itin: list[int], P: list[int]
) -> list:
    """Recursively generates an itinerary of length 'length'"""

    actual_itin = [n[1] for n in itin if n[1] is not None]

    if len(actual_itin) == length:
        if actual_itin in P:
            return itinerary_builder(graph, length, [], P)
        return actual_itin

    if not itin:
        neighbours = []
        while not neighbours:
            neighbours = graph.get_neighbours(random.choice(graph.get_all_nodes()))
        itin.append(random.choice(neighbours))
    else:
        next_options = graph.get_neighbours(itin[-1][0])
        if not next_options:  # Try to find another way
            return itinerary_builder(graph, length, [], P)
        for option in next_options:
            if itin[-1][1] != option[1]:
                itin.append(option)
                break
        else:
            return itinerary_builder(graph, length, [], P)

    return itinerary_builder(graph, length, itin, P)


def generate_itineraries(graph: AdjanecyList, itin_classes: dict[int, int]) -> list:
    P = []

    for length, num_itins in itin_classes.items():
        for _ in range(num_itins):
            P.append(itinerary_builder(graph, length, [], P))

    return P