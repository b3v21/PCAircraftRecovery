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

def calculate_time(loc1 : tuple[float, float], loc2 : tuple[float, float]) -> float:
    """
    Calculates the time it takes to fly between two locations
    """
    
    # Time is distance in km / average speed of 700km/h + 30min for takeoff/landing
    return round(geopy.distance.geodesic(loc1,loc2).km / 700 + 0.5,1)

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
    departure_time = round(random.random() * TIME_HORIZON - ceil(flight_time), 1) # TODO: this needs to be changed when flight times are changed
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


def create_graph(num_flights, flight_distribution):
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

################################# RUN ENGINE WITH DATA #################################

random.seed(690)
num_flights = floor(random.normalvariate(50, 5))
flight_distribution = divide_number(
    num_flights, len(AIRPORTS), 2, 30
)

graph = create_graph(num_flights, flight_distribution)
for node, neighs in graph.adj_list.items():
    print(node, neighs)

num_flights = graph.count_all_flights()
num_tails = graph.count_all_flights()  # This is somewhat arbitrary
num_airports = 10
num_fare_classes = 2  # This is somewhat arbitrary
num_delay_levels = 2  # This is somewhat arbitrary

# Sets
T = range(num_tails)
F = range(num_flights)
K = AIRPORTS
Y = range(num_fare_classes)
Z = range(num_delay_levels)

# This represents the different types of itineraries which will be generated, in this case
# there are 5 itineraries of length 1, 3 of length 2 and 2 of length 3. NOTE: if a user
# tries to generate an itinerary which is too long, a maximum recusion depth error will occur.

itin_classes = {1: 20, 2: 7, 3: 5, 4:1}

try:
    P = generate_itineraries(graph, itin_classes)
except RecursionError:
    print("ERROR: Recursion depth exceeded, please reduce itinerary length")

print(f"There are {num_flights} flights")
[print(p) for p in P]

# Construct arrival and departure times
# NOTE: THESE ARE LISTS IN PREV DATA FILES, NOW DICTS
std = {}
sta = {}
for n in graph.adj_list.keys():
    for neigh, flight_id in graph.get_neighbours(n):
        if flight_id != None:
            std[flight_id] = n.time
            sta[flight_id] = neigh.time

# Construct arrival and departure slots
DA = [(float(t), float(t + 0.25)) for t in np.arange(0, TIME_HORIZON, 0.25)]
AA = [(float(t), float(t + 0.25)) for t in np.arange(0, TIME_HORIZON, 0.25)]

# Set of arrival and departure slots compatible with flight f (dict indexed by flight)
AAF = {
    f: [i for i, slot in enumerate(AA) if sta[f] <= slot[1] and sta[f] >= slot[0]]
    for f in F
}
DAF = {
    f: [i for i, slot in enumerate(DA) if std[f] <= slot[1] and std[f] >= slot[0]]
    for f in F
}

# Set of flights compatible with arrive/departure slot asl/dsl (dict index by asl/dsl)
FAA = {asl: [f for f in F if sta[f] <= asl[1] and sta[f] >= asl[0]] for asl in AA}
FDA = {dsl: [f for f in F if std[f] <= dsl[1] and std[f] >= dsl[0]] for dsl in DA}

# set of flights compatible with tail T
# (currently every flight is compatible with every tail)
F_t = {t: list(F) for t in T}

# set of tails compatible with flight F
T_f = {f: [t for t in T if f in F_t[t]] for f in F}

# Set of flights f which arrive to airport k
FA_k = {k: [] for k in K}

for dep_node in graph.adj_list.keys():
    arr_nodes = graph.get_neighbours(dep_node)
    for node in arr_nodes:
        if node[1] is not None:
            FA_k[node[0].get_name()] += [node[1]]   
        
# Set of flights f which depart from airport k
FD_k = {k: [] for k in K}

for k in K:
    airport_nodes = graph.get_nodes(k)
    for node in airport_nodes:
        FD_k[k] += [f for f in F if f in list(zip(*graph.get_neighbours(node)))[1]]

# THESE ARENT USED IN THE ACTUAL MODEL, JUST USED TO PRODUCE DATA BELOW
DK_f = {}
for airport, flights in FD_k.items():
    for flight in flights:
        DK_f[flight] = airport

AK_f = {}
for airport, flights in FA_k.items():
    for flight in flights:
        AK_f[flight] = airport

# Set of flights fd compatible with a connection from flight f
# fd is compatible if it is scheduled to depart from the arrival airport of flight f
# and the scheduled arrival of f is before the scheduled departure of fd
CF_f = {
    f: [fd for fd in F if AK_f[f] == DK_f[fd] and sta[f] <= std[fd] and fd != f]
    for f in F
}

# Subset of itineraries compatible with a reassignment from an original itinerary p.
# itinary p is compatible for a reassignment with itinary pd if they both share the
# same start and end destination
CO_p = {
    P.index(p): [
        P.index(pd)
        for pd in P
        if pd != [] and DK_f[pd[0]] == DK_f[p[0]] and AK_f[pd[-1]] == AK_f[p[-1]]
    ]
    for p in P
    if p != []
}


# Cost of operating flight f with tail t
oc = {(t, f): 1500 for t in T for f in F}

# Delay cost per minute of arrival delay of flight f
dc = {f: 100 for f in F}

# Number of passengers in fare class v that are originally scheduled to
# take itinerary p
n = {(v, P.index(p)): 25 for v in Y for p in P}

# Seating capacity of tail t in T
q = {t: 250 for t in T}

# Reaccommodation Cost for a passenger reassigned from p to pd.
rc = {
    (P.index(p), P.index(pd)): (lambda p, pd: 0 if p == pd else 0.5)(p, pd)
    for p in P
    for pd in P
}

# Phantom rate for passenger in fare class v reassigned from p to pd with delay level
# zeta
theta = {
    (y, P.index(p), P.index(pd), z): 0 for y in Y for p in P for pd in P for z in Z
}

# Capacity of arrival and departure slots
scA = {asl: len(AIRPORTS) for asl in AA}
scD = {dsl: len(AIRPORTS) for dsl in DA}

# Scheduled buffer time for each flight (set to 0 for now)
sb = {f: 0 for f in F}

# minimum turn time between flight f and fd with tail t
mtt = {(t, f, fd): 0 for t in T for f in F for fd in F}

# minimum connection time between flight f and fd in itinerary p
mct = {(P.index(p), f, fd): 0 for p in P for f in F for fd in F}

# Planned connection time between flights f and fd. It equals scheduled departure time of
# flight fd minus the scheduled arrival time of flight f.
ct = {(f, fd): max(0, std[fd] - sta[f]) for fd in F for f in F}

# set of ordered flight pairs of consecutive flights in itinary p.
CF_p = {
    P.index(p): [(p[i], p[i + 1]) for i, _ in enumerate(p[:-1])]
    for p in P
}

# One if flight f is the last flight of itinerary p, and zero otherwise.
lf = {
    (P.index(p), f): (lambda last: 1 if last == f else 0)(p[-1]) for p in P for f in F
}

# Upper bound on the delay, expressed in minutes, corresponding to delay level ζ.
small_theta = {z: 1000 for z in Z}

# Extra fuel cost for delay absorption (through cruise speed increases) per minute for
# flight f.
fc = {f: 100 for f in F}

# Sum of the cost of the loss of goodwill and the compensation cost (if any) for a
# passenger who was scheduled to take itinerary p and is reassigned to itinerary p’, if
# the passenger’s destination arrival delay via itinerary p′ compared with the planned
# arrival time of itinerary p corresponds to delay level ζ
pc = {(z, P.index(p), P.index(pd)): 0 for z in Z for p in P for pd in P}

# Per-flight schedule change penalty for not operating the flight using the originally
# planned tail.
kappa = 100

# One if flight f was originally scheduled to be operated by tail t, and zero otherwise.
for node in graph.adj_list.keys():
    for neigh, flight_id in graph.get_neighbours(node):
        if flight_id != None:
            x_hat = {(f, t): 1 if t == f else 0 for f in F for t in T}

# Starting location of planes (binary)
tb = {(t, k): 0 for t in T for k in K}
tail_count = 0
for flight, airport in DK_f.items():
    tb[flight, airport] = 1
