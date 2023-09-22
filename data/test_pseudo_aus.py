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

random.seed(69)

TIME_HORIZON = 72


class Node:
    def __init__(self, name, lat, long, time):
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

    def new_time_copy(self, time):
        new_node = copy.deepcopy(self)
        new_node.time = time
        return new_node


class AdjanecyList:
    def __init__(self):
        self.adj_list = {}

    def add_node(self, node, neighbours):
        self.adj_list[node] = []
        for neighbour in neighbours:
            self.adj_list[node].append(neighbour)

    def get_node(self, name):
        for node in self.adj_list:
            if node.name == name:
                return node
        return None

    def get_neighbours(self, node):
        return self.adj_list[node]

    def get_all_nodes(self):
        return self.adj_list.keys()
    
    def count_node_locations(self):
        counts = {"SYD":0, "MEL":0, "BNE":0, "PER":0, "ADL":0, "OOL":0, "CNS":0, "HBA":0, "CBR":0, "TSV":0}
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
        Node("SYD", -33.949995, 151.181705, None),
        Node("BNE", -27.386944, 153.1175, None),
        Node("MEL", -37.673333, 144.843333, None),
        Node("PER", -31.940278, 115.966944, None),
        Node("ADL", -34.945, 138.530556, None),
        Node("OOL", -28.164444, 153.505278, None),
        Node("CNS", -16.885833, 145.755278, None),
        Node("HBA", -42.836111, 147.510278, None),
        Node("CBR", -35.306944, 149.195, None),
        Node("TSV", -19.2525, 146.765278, None),
    ]

    adj_list = AdjanecyList()
    flights_remaining = int(random.normalvariate(1000, 100))
    
    for node in default_nodes:
        if node.name == "SYD":
            flights_used = random.randrange(int(0.5*flights_remaining), int(0.81*flights_remaining))
        else:
            flights_used = random.randrange(0, int(0.3 * flights_remaining))
        
        for _ in range(0, flights_used):
            neighbours = []
            
            # Randomise destination airport
            dest_node = random.choice(default_nodes)
            while dest_node == node:
                dest_node = random.choice(default_nodes)

            # Randomise time flight is scheduled to depart
            departure_time = round(random.random() * TIME_HORIZON, 1)
            departure_node = node.new_time_copy(departure_time)
            
            neighbours.append(dest_node.new_time_copy(departure_time + 2))
            
            adj_list.add_node(departure_node, neighbours)

        flights_remaining -= flights_used

    return adj_list

graph = create_graph()
print(graph.count_node_locations())

