import random
import numpy as np
from math import floor
from .build_psuedo_aus import *

"""
This is a simple testcase for the psuedo_aus model.
"""

random.seed(69)
num_flights = floor(random.normalvariate(100, 20))
flight_distribution = divide_number(num_flights, len(AIRPORTS), 0.25, 0.35)

graph = create_graph(flight_distribution)

print("graph created")

num_flights = graph.count_all_flights()
num_tails = num_flights  # This is somewhat arbitrary
num_airports = 10
num_fare_classes = 2  # This is somewhat arbitrary
num_delay_levels = 5  # This is somewhat arbitrary

# Sets
T = range(num_tails)
F = range(num_flights)
K = AIRPORTS
Y = range(num_fare_classes)
Z = range(num_delay_levels)

# This represents the different types of itineraries which will be generated, in this case
# there are 5 itineraries of length 1, 3 of length 2 and 2 of length 3. NOTE: if a user
# tries to generate an itinerary which is too long, a maximum recusion depth error will occur.

itin_classes = {1: num_flights}

# try:
#     P = generate_itineraries(graph, itin_classes)
# except RecursionError:
#     print("ERROR: Recursion depth exceeded, please reduce itinerary length")

P = [[p] for p in range(5)]

print("itineraries created")

# DEBUG GRAPH PRINTS
# for node, neigh in graph.adj_list.items():
#     print(node, ": ", [n for n in neigh if n[1] is not None])

# Construct arrival and departure times
std = {}
sta = {}
for n in graph.adj_list.keys():
    for neigh, flight_id in graph.get_neighbours(n):
        if flight_id != None:
            std[flight_id] = n.time
            sta[flight_id] = neigh.time

# Construct arrival and departure slots
DA = [(float(t), float(t + 2)) for t in np.arange(0, TIME_HORIZON, 2)]
AA = [(float(t), float(t + 2)) for t in np.arange(0, TIME_HORIZON, 2)]

# Set of arrival and departure slots compatible with flight f (dict indexed by flight)
AAF = {f: [i for i, slot in enumerate(AA) if sta[f] <= slot[0]] for f in F}
DAF = {f: [i for i, slot in enumerate(DA) if std[f] <= slot[0]] for f in F}

# Set of flights compatible with arrive/departure slot asl/dsl (dict index by asl/dsl)
FAA = {asl: [f for f in F if sta[f] <= asl[0]] for asl in AA}
FDA = {dsl: [f for f in F if std[f] <= dsl[0]] for dsl in DA}

# Capacity of arrival and departure slots
scA = {asl: 100 for asl in AA}
scD = {dsl: 100 for dsl in DA}

print("slot data created")

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

# set of ordered flight pairs of consecutive flights in itinary p.
CF_p = {P.index(p): [(p[i], p[i + 1]) for i, _ in enumerate(p[:-1])] for p in P}

# Planned connection time between flights f and fd. It equals scheduled departure time of
# flight fd minus the scheduled arrival time of flight f.
ct = {(f, fd): max(0, std[fd] - sta[f]) for fd in F for f in F}

# One if flight f is the last flight of itinerary p, and zero otherwise.
lf = {
    (P.index(p), f): (lambda last: 1 if last == f else 0)(p[-1]) for p in P for f in F
}

print("itinerary and flight data created")


# Cost of operating flight f with tail t
oc = {(t, f): 10000 for t in T for f in F}

# Delay cost per hour of arrival delay of flight f
dc = {f: 12500 for f in F}

# Number of passengers in fare class v that are originally scheduled to
# take itinerary p
n = {(v, P.index(p)): 25 for v in Y for p in P}

# Seating capacity of tail t in T
q = {t: 10000000 for t in T}

# Reaccommodation Cost for a passenger reassigned from p to pd.
rc = {
    (P.index(p), P.index(pd)): (lambda p, pd: 0 if p == pd else 800)(p, pd)
    for p in P
    for pd in P
}

# Phantom rate for passenger in fare class v reassigned from p to pd with delay level
# zeta
phantom_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

theta = {
    (y, P.index(p), P.index(pd), z): phantom_rates[z]
    for y in Y
    for p in P
    for pd in P
    for z in Z
}

# Scheduled buffer time for each flight (set to 0 for now)
sb = {f: 0 for f in F}

# minimum turn time between flight f and fd with tail t
mtt = 1 # FIX THIS SHIT ASAP

# minimum connection time between flight f and fd in itinerary p
# mct = {(P.index(p), f, fd): 0 for p in P for f in F for fd in F}
mct = 1 # Hardcoded for this example to reduce data creation time
# FIX THIS ALSO


# Upper bound on the delay, expressed in hours, corresponding to delay level ζ.
small_theta = {0: 1, 1: 2, 2: 5, 3: 10, 4: 72}


# Extra fuel cost for delay absorption (through cruise speed increases) per hour for
# flight f.
fc = {f: 30000 for f in F}

# Sum of the cost of the loss of goodwill and the compensation cost (if any) for a
# passenger who was scheduled to take itinerary p and is reassigned to itinerary p’, if
# the passenger’s destination arrival delay via itinerary p′ compared with the planned
# arrival time of itinerary p corresponds to delay level ζ
pc = {(z, P.index(p), P.index(pd)): 250 for z in Z for p in P for pd in P}


print("cost data created")


# Per-flight schedule change penalty for not operating the flight using the originally
# planned tail.
kappa = 0  # UNBOUNDED FOR NOW TO REMOVE X_HAT CONTRIBUTION

# Starting location of planes (binary)
tb = {(t, k): 0 for t in T for k in K}

# One if flight f was originally scheduled to be operated by tail t, and zero otherwise.
x_hat = {(f, t): 0 for f in F for t in T}

P_sorted = sorted(P, key=(lambda x : std[x[0]]))

tail_count = 0

for itin in P_sorted:
    airport = DK_f[itin[0]]
    tb[(tail_count, airport)] = 1
    tail_count += 1
    if tail_count == num_tails:
        break
    
print([(DK_f[p[0]], AK_f[p[0]], std[p[0]], sta[p[0]]) for p in P])
print([x for x in tb if tb[x] == 1])

print("remaining data created")
