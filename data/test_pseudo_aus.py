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

#########################################################################################
# NOTE: THIS IS WRONG BUT FOR NOW EVERY FLIGHT IS 2HRS LONG
#########################################################################################

import copy
import random
import numpy as np

random.seed(69)

TIME_HORIZON = 72


class Node:
    def __init__(self, node_id, name, lat, long, time):
        self.id = node_id
        self.name = name
        self.lat = lat
        self.long = long
        self.time = time

    def __repr__(self):
        return "Node({},{},{},{})".format(self.name, self.lat, self.long, self.time)

    def __str__(self):
        return "Node({},{},{},{})".format(self.name, self.lat, self.long, self.time)

    def __eq__(self, __value: object) -> bool:
        if self.name == __value.name and self.time == __value.time:
            return True

    def __hash__(self) -> int:
        return hash(str(self))

    def get_id(self):
        return self.id

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

    def add_node(self, node, neighbours, flight_id):
        self.adj_list[node] = []
        for neighbour in neighbours:
            self.adj_list[node].append((neighbour, flight_id))

    def get_node(self, name):
        for node in self.adj_list:
            if node.name == name:
                return node
        return None

    def add_to_node(self, name, neighbour, flight_id):
        self.adj_list[name].append((neighbour, flight_id))

    def get_neighbours(self, node):
        return self.adj_list[node]

    def get_all_nodes(self):
        return self.adj_list.keys()

    def count_node_locations(self):
        counts = {
            "SYD": 0,
            "MEL": 0,
            "BNE": 0,
            "PER": 0,
            "ADL": 0,
            "OOL": 0,
            "CNS": 0,
            "HBA": 0,
            "CBR": 0,
            "TSV": 0,
        }
        for node in self.adj_list.keys():
            counts[node.name] += 1
        return counts

    def __repr__(self):
        output = ""
        for node, neighbours in self.adj_list.items():
            output += "{}: {}\n".format(node, str(neighbours))
        return output


def create_graph():
    # Create default nodes
    default_nodes = [
        Node(0, "SYD", -33.949995, 151.181705, None),
        Node(0, "BNE", -27.386944, 153.1175, None),
        Node(0, "MEL", -37.673333, 144.843333, None),
        Node(0, "PER", -31.940278, 115.966944, None),
        Node(0, "ADL", -34.945, 138.530556, None),
        Node(0, "OOL", -28.164444, 153.505278, None),
        Node(0, "CNS", -16.885833, 145.755278, None),
        Node(0, "HBA", -42.836111, 147.510278, None),
        Node(0, "CBR", -35.306944, 149.195, None),
        Node(0, "TSV", -19.2525, 146.765278, None),
    ]

    current_node_id = 0
    current_flight_id = 0
    adj_list = AdjanecyList()
    flights_remaining = int(random.normalvariate(20, 8))

    for node in default_nodes:
        if node.name == "SYD":
            flights_used = random.randrange(
                int(0.4 * flights_remaining), int(0.61 * flights_remaining)
            )
        else:
            flights_used = random.randrange(0, int(0.3 * flights_remaining))

        for _ in range(0, flights_used):
            neighbours = []

            # Randomise destination airport
            dest_node = random.choices(
                default_nodes,
                weights=[0.25, 0.15, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            )[0]
            while dest_node == node:
                dest_node = random.choices(
                    default_nodes,
                    weights=[0.25, 0.15, 0.15, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                )[0]

            # Randomise time flight is scheduled to depart
            departure_time = round(random.random() * TIME_HORIZON, 1)
            departure_node = node.new_time_copy(departure_time, current_node_id)

            if departure_node in adj_list.get_all_nodes():
                adj_list.add_to_node(
                    departure_node,
                    dest_node.new_time_copy(departure_time + 2, current_node_id + 1),
                    current_flight_id,
                )
            else:
                neighbours.append(
                    dest_node.new_time_copy(departure_time + 2, current_node_id + 1)
                )
                adj_list.add_node(departure_node, neighbours, current_flight_id)

            current_node_id += 1
            current_flight_id += 1
        flights_remaining -= flights_used

    return adj_list


def extract_data(graph: AdjanecyList) -> None:
    num_flights = sum(graph.count_node_locations().values())
    num_tails = 20  # This is somewhat arbitrary (up to us to decide size of fleet)
    num_airports = 10
    num_fare_classes = 2  # This is somewhat arbitrary
    num_delay_levels = 2  # This is somewhat arbitrary

    # Sets
    T = range(num_tails)
    F = range(num_flights)
    K = range(num_airports)
    Y = range(num_fare_classes)
    Z = range(num_delay_levels)

    # Set of itineraries, this needs to be reworked for multi-leg itins
    P = []
    for n in graph.get_all_nodes():
        for neigh, flight_id in graph.get_neighbours(n):
            P.append([flight_id])

    # Construct arrival and departure times
    # NOTE: THESE ARE LISTS IN PREV DATA FILES, NOW DICTS
    std = {}
    sta = {}
    for n in graph.get_all_nodes():
        for neigh, flight_id in graph.get_neighbours(n):
            std[flight_id] = n.time
            sta[flight_id] = neigh.time

    # Construct arrival and departure slots
    DA = [(t, t + 0.25) for t in np.arange(0, TIME_HORIZON, 0.25)]
    AA = [(t, t + 0.25) for t in np.arange(0, TIME_HORIZON, 0.25)]


if __name__ == "__main__":
    graph = create_graph()
    print(graph.count_node_locations())
    extract_data(graph)
