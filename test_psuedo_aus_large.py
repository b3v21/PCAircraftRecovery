from gurobipy import *
from airline_recovery import *
from data.build_psuedo_aus import *
import random
import numpy as np
from math import floor

random.seed(59)
graph_nodes = floor(random.normalvariate(1000, 100))
flight_distribution = divide_number(graph_nodes, len(AIRPORTS), 0.25, 0.35)

graph = create_graph(flight_distribution)
num_flights = graph.count_all_flights()
print("graph created")

# This represents the different types of itineraries which will be generated
singles = []
for p in range(num_flights):
    singles.append([p])

itin_classes = {}
try:
    P = generate_itineraries(graph, itin_classes, singles)
except RecursionError:
    print("ERROR: Recursion depth exceeded, please reduce itinerary length")

P.insert(0, [])

print([p for p in P if len(p) > 1])
print("itineraries created")


def build_base_data() -> tuple:
    num_tails = 123  # This is somewhat arbitrary
    num_airports = 10
    num_fare_classes = 4  # This is somewhat arbitrary
    num_delay_levels = 5  # This is somewhat arbitrary

    # Sets
    T = range(num_tails)
    F = range(num_flights)
    K = AIRPORTS
    Y = range(num_fare_classes)
    Z = range(num_delay_levels)

    # DEBUG GRAPH PRINTS
    for node, neigh in graph.adj_list.items():
        print(node, ": ", [n for n in neigh if n[1] is not None])

    # Construct arrival and departure times
    std = {}
    sta = {}
    for n in graph.adj_list.keys():
        for neigh, flight_id in graph.get_neighbours(n):
            if flight_id != None:
                std[flight_id] = n.time
                sta[flight_id] = neigh.time

    # Construct arrival and departure slots
    DA = [(float(t), float(t + 0.5)) for t in np.arange(0, TIME_HORIZON, 0.5)]
    AA = [(float(t), float(t + 0.5)) for t in np.arange(0, TIME_HORIZON, 0.5)]

    # Set of arrival and departure slots compatible with flight f (dict indexed by flight)
    AAF = {f: [i for i, slot in enumerate(AA) if sta[f] <= slot[0]] for f in F}
    DAF = {f: [i for i, slot in enumerate(DA) if std[f] <= slot[0]] for f in F}

    # Set of flights compatible with arrive/departure slot asl/dsl (dict index by asl/dsl)
    FAA = {asl: [f for f in F if sta[f] <= asl[0]] for asl in AA}
    FDA = {dsl: [f for f in F if std[f] <= dsl[0]] for dsl in DA}

    # Capacity of arrival and departure slots
    scA = {asl: 20 for asl in AA}
    scD = {dsl: 20 for dsl in DA}

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
            if pd != []
            and DK_f[pd[0]] == DK_f[p[0]]
            and AK_f[pd[-1]] == AK_f[p[-1]]
            and std[pd[0]] >= std[p[0]]
        ]
        for p in P
        if p != []
    }

    # Manually define the the compatibility of the empty itinerary
    CO_p[0] = [0]

    # Manually add the empty itinerary as a compatible itinerary for each itinerary
    for p in P:
        CO_p[P.index(p)].append(0)

    # set of ordered flight pairs of consecutive flights in itinary p.
    CF_p = {P.index(p): [(p[i], p[i + 1]) for i, _ in enumerate(p[:-1])] for p in P}

    # Planned connection time between flights f and fd. It equals scheduled departure time of
    # flight fd minus the scheduled arrival time of flight f.
    ct = {(f, fd): max(0, std[fd] - sta[f]) for fd in F for f in F}

    # One if flight f is the last flight of itinerary p, and zero otherwise.
    lf = {
        (P.index(p), f): (lambda last: 1 if last == f else 0)(p[-1] if p != [] else 0)
        for p in P
        for f in F
    }
    
    tail_cap = {"Boeing 737-800": 174, "Boeing 787-9" : 236, "Airbus A380-800" : 485, "Airbus A330-300" : 297, "Airbus A330-200" : 271}
    tail_amount = {"Boeing 737-800": 75, "Boeing 787-9" : 14, "Airbus A380-800" : 10, "Airbus A330-300" : 10, "Airbus A330-200" : 14}

    q = {}
    # Seating capacity of tail t in T
    for t in T:
        tail = random.choice(list(tail_cap.keys()))
        while tail_amount[tail] <= 0:
            tail = random.choice(list(tail_cap.keys()))
        tail_amount[tail] -= 1
        q[t] = tail_cap[tail]

    print("itinerary and flight data created")

    # Cost of operating flight f with tail t
    oc = {(t, f): 10000 for t in T for f in F}

    # Delay cost per hour of arrival delay of flight f
    dc = {f: 12500 for f in F}

    # Number of passengers in fare class v that are originally scheduled to
    # take itinerary p
    n = {(v, P.index(p)): 50 for v in Y for p in P}

    # Reaccommodation Cost for a passenger reassigned from p to pd.
    rc = {
        (P.index(p), P.index(pd)): (lambda p, pd: 0 if p == pd else 600)(p, pd)
        for p in P
        for pd in P
    }

    for p in P:
        rc[(P.index(p), 0)] = 1500

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
    mtt = {(t, f, fd): 1 for t in T for f in F for fd in F}

    # minimum connection time between flight f and fd in itinerary p
    mct = {(P.index(p), f, fd): 1 for p in P for f in F for fd in F}

    # Upper bound on the delay, expressed in hours, corresponding to delay level ζ.
    small_theta = {0: 1, 1: 2, 2: 5, 3: 10, 4: 72}

    # Extra fuel cost for delay absorption (through cruise speed increases) per hour for
    # flight f.
    fc = {f: 30000 for f in F}

    # Sum of the cost of the loss of goodwill and the compensation cost (if any) for a
    # passenger who was scheduled to take itinerary p and is reassigned to itinerary p’, if
    # the passenger’s destination arrival delay via itinerary p′ compared with the planned
    # arrival time of itinerary p corresponds to delay level ζ
    pc = {(z, P.index(p), P.index(pd)): 200 + z * 150 for z in Z for p in P for pd in P}

    print("cost data created")

    # Per-flight schedule change penalty for not operating the flight using the originally
    # planned tail.
    kappa = 0  # UNBOUNDED FOR NOW TO REMOVE X_HAT CONTRIBUTION

    # Starting location of planes (binary)
    tb = {(t, k): 0 for t in T for k in K}

    # One if flight f was originally scheduled to be operated by tail t, and zero otherwise.
    x_hat = {(f, t): 0 for f in F for t in T}

    P_sorted = sorted(P, key=(lambda x: std[x[0]] if x != [] else 0))
    tail_count = 0

    for itin in P_sorted:
        if itin:
            airport = DK_f[itin[0]]
            tb[(tail_count, airport)] = 1
            tail_count += 1
            if tail_count == num_tails:
                break

    print("remaining data created")

    return (
        T,
        F,
        K,
        Y,
        Z,
        P,
        sta,
        std,
        AA,
        DA,
        AAF,
        DAF,
        FAA,
        FDA,
        F_t,
        T_f,
        FA_k,
        FD_k,
        DK_f,
        AK_f,
        CF_f,
        CO_p,
        oc,
        dc,
        n,
        q,
        rc,
        theta,
        scA,
        scD,
        sb,
        mtt,
        mct,
        ct,
        CF_p,
        lf,
        small_theta,
        fc,
        pc,
        kappa,
        x_hat,
        tb,
    )


def test_psuedo_aus_large_size():
    m = Model("airline recovery aus large")

    (
        T,
        F,
        K,
        Y,
        Z,
        P,
        sta,
        std,
        AA,
        DA,
        AAF,
        DAF,
        FAA,
        FDA,
        F_t,
        T_f,
        FA_k,
        FD_k,
        DK_f,
        AK_f,
        CF_f,
        CO_p,
        oc,
        dc,
        n,
        q,
        rc,
        theta,
        scA,
        scD,
        sb,
        mtt,
        mct,
        ct,
        CF_p,
        lf,
        small_theta,
        fc,
        pc,
        kappa,
        x_hat,
        tb,
    ) = build_base_data()

    variables = generate_variables(m, T, F, Y, Z, P, AA, DA, CO_p)
    set_objective(
        m, variables, T, F, Y, Z, P, F_t, CO_p, oc, dc, rc, theta, fc, pc, kappa, x_hat
    )
    # flight_scheduling_constraints(m, variables, F, T_f)
    sequencing_and_fleet_size_constraints(
        m, variables, T, F, K, F_t, T_f, FD_k, CF_f, tb
    )
    passenger_flow_constraints(m, variables, F, Y, Z, P, T_f, CO_p, theta, n, q)
    airport_slot_constraints(
        m, variables, F, Z, sta, std, AA, DA, AAF, DAF, FAA, FDA, scA, scD
    )
    flight_delay_constraints(m, variables, T, F, T_f, CF_f, sb, mtt, ct)
    itinerary_feasibility_constraints(m, variables, F, P, sta, std, CF_p, mct)
    itinerary_delay_constraints(m, variables, F, Z, P, sta, CO_p, lf, small_theta)
    beta_linearizing_constraints(m, variables, Y, Z, P, CO_p)

    print("optimizing to get xhat...")
    m.setParam("OutputFlag", 1)
    m.setParam("MIPGap", 0.1)
    m.optimize()

    x_hat = generate_x_hat(m, variables, F, T)

    kappa = 1000

    # Generate new data
    AAF = {f: [i for i, slot in enumerate(AA) if sta[f] <= slot[0]] for f in F}
    DAF = {f: [i for i, slot in enumerate(DA) if std[f] <= slot[0]] for f in F}

    FAA = {asl: [f for f in F if sta[f] <= asl[0]] for asl in AA}
    FDA = {dsl: [f for f in F if std[f] <= dsl[0]] for dsl in DA}

    print("Regenerate neccecary constraints...")
    airport_slot_constraints(
        m, variables, F, Z, sta, std, AA, DA, AAF, DAF, FAA, FDA, scA, scD
    )
    itinerary_feasibility_constraints(m, variables, F, P, sta, std, CF_p, mct)
    itinerary_delay_constraints(m, variables, F, Z, P, sta, CO_p, lf, small_theta)
    set_objective(
        m, variables, T, F, Y, Z, P, F_t, CO_p, oc, dc, rc, theta, fc, pc, kappa, x_hat
    )

    print("optimizing...")
    m.setParam("OutputFlag", 1)
    m.setParam("MIPGap", 0.1)
    m.optimize()

    print("generating output...")
    generate_output(
        m, variables, T, F, Y, Z, P, sta, std, AA, DA, DK_f, AK_f, CF_f, n, fc
    )
