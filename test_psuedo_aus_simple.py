import random
import numpy as np
from math import floor
from data.build_psuedo_aus import *
from gurobipy import *
from airline_recovery import *

"""
This is a simple testcase for the psuedo_aus model.
"""

random.seed(69)
num_flights = floor(random.normalvariate(10, 1))
flight_distribution = divide_number(num_flights, len(AIRPORTS), 0.20, 0.3)

graph = create_graph(flight_distribution)


num_flights = graph.count_all_flights()
num_tails = 3  # This is somewhat arbitrary
num_airports = 10
num_fare_classes = 2  # This is somewhat arbitrary
num_delay_levels = 2  # This is somewhat arbitrary

# Sets
T = range(num_tails)
F = range(num_flights)
K = AIRPORTS
Y = range(num_fare_classes)
Z = range(num_delay_levels)

T_m = set()

# This represents the different types of itineraries which will be generated, in this case
# there are 5 itineraries of length 1, 3 of length 2 and 2 of length 3. NOTE: if a user
# tries to generate an itinerary which is too long, a maximum recusion depth error will occur.

itin_classes = {1: 4, 2: 1}

try:
    P = generate_itineraries(graph, itin_classes, [])
except RecursionError:
    print("ERROR: Recursion depth exceeded, please reduce itinerary length")

# DEBUG GRAPH PRINTS
for node in graph.adj_list.items():
    print(node)
print(P)

# Construct arrival and departure times
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
        if pd != []
        and DK_f[pd[0]] == DK_f[p[0]]
        and AK_f[pd[-1]] == AK_f[p[-1]]
        and std[pd[0]] >= std[p[0]]
        and sta[pd[-1]] >= sta[p[-1]]
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
n = {(v, P.index(p)): 20 for v in Y for p in P}

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
CF_p = {P.index(p): [(p[i], p[i + 1]) for i, _ in enumerate(p[:-1])] for p in P}

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
x_hat = {(f, t): 0 for t in T for f in F}
x_hat[(6, 0)] = 1
x_hat[(2, 1)]
x_hat[(1, 2)]

# Starting location of planes (binary)
tb = {(t, k): 0 for t in T for k in K}
tb[(0, "PER")] = 1
tb[(1, "SYD")] = 1
tb[(2, "SYD")] = 1


def test_simple_limited_tails():
    limited_tails = Model("limited_tails_test")

    variables = generate_variables(limited_tails, T, F, Y, Z, P, AA, DA, CO_p, K)
    set_objective(
        limited_tails,
        variables,
        T,
        F,
        Y,
        Z,
        P,
        F_t,
        CO_p,
        oc,
        dc,
        rc,
        theta,
        fc,
        pc,
        kappa,
        x_hat,
    )
    flight_scheduling_constraints(limited_tails, variables, F, T_f)
    sequencing_and_fleet_size_constraints(
        limited_tails, variables, T, F, K, F_t, T_f, FD_k, CF_f, tb
    )
    passenger_flow_constraints(
        limited_tails, variables, F, Y, Z, P, T_f, CO_p, theta, n, q
    )
    airport_slot_constraints(
        limited_tails, variables, F, Z, sta, std, AA, DA, AAF, DAF, FAA, FDA, scA, scD
    )
    flight_delay_constraints(limited_tails, variables, T, F, T_f, CF_f, sb, mtt, ct)
    itinerary_feasibility_constraints(
        limited_tails, variables, F, P, sta, std, CF_p, mct
    )
    itinerary_delay_constraints(
        limited_tails, variables, F, Z, P, sta, CO_p, lf, small_theta
    )
    beta_linearizing_constraints(limited_tails, variables, Y, Z, P, CO_p)

    print("optimizing...")
    limited_tails.optimize()

    print("generating output...")
    generate_output(
        limited_tails,
        variables,
        T,
        F,
        Y,
        Z,
        P,
        sta,
        std,
        AA,
        DA,
        DK_f,
        AK_f,
        CF_f,
        n,
        fc,
        T_m,
        F_t,
        oc,
        dc,
        CO_p,
        rc,
        theta,
        pc,
        kappa,
        x_hat,
    )

    (
        x,
        z,
        y,
        sigma,
        rho,
        phi,
        h,
        lambd,
        alpha,
        deltaA,
        deltaD,
        vA,
        vD,
        gamma,
        tao,
        beta,
        imt,
        fmt,
        w,
        sigma_m,
        rho_m,
        m_t,
        m_m,
        phi_m,
    ) = variables

    # Cancellations occur as expected
    for f in F:
        if f == 0 or f == 4 or f == 5:
            assert z[f].x > 0.9
        else:
            assert z[f].x < 0.9
    for p in P:
        assert lambd[P.index(p)].x < 0.9

    # Flights assigned to correct tails (even multiflight tails)
    assert x[2, 1].x > 0.9
    assert x[1, 2].x > 0.9
    assert x[0, 6].x > 0.9
    assert x[0, 3].x > 0.9
