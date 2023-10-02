from gurobipy import *
from airline_recovery import *
from data.build_psuedo_aus import *
import random
import numpy as np
from math import floor

random.seed(59)

num_flights = floor(random.normalvariate(20, 1))
flight_distribution = divide_number(num_flights, len(AIRPORTS), 0.25, 0.35)
graph = create_graph(flight_distribution)
print("graph created")

itin_classes = {1: 20, 2: 5, 3: 1}

try:
    P = generate_itineraries(graph, itin_classes, [])
except RecursionError:
    print("ERROR: Recursion depth exceeded, please reduce itinerary length")

print("\nitineraries created")
print(P, "\n")

print("Graph")
for node, neigh in graph.adj_list.items():
    print(node, ": ", [n for n in neigh if n[1] is not None])
print()


def build_base_data() -> tuple:
    num_flights = graph.count_all_flights()
    num_tails = 30
    num_airports = 10
    num_fare_classes = 2
    num_delay_levels = 5

    # Sets
    T = range(num_tails)
    F = range(num_flights)
    K = AIRPORTS
    Y = range(num_fare_classes)
    Z = range(num_delay_levels)

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
    oc = {(t, f): 10000 for t in T for f in F}

    # Delay cost per hour of arrival delay of flight f
    dc = {f: 12500 for f in F}

    # Number of passengers in fare class v that are originally scheduled to take itinerary p
    n = {(v, P.index(p)): 50 for v in Y for p in P}

    # Seating capacity of tail t in T
    q = {t: 250 for t in T}

    # Reaccommodation Cost for a passenger reassigned from p to pd.
    rc = {
        (P.index(p), P.index(pd)): (lambda p, pd: 0 if p == pd else 800)(p, pd)
        for p in P
        for pd in P
    }

    # Phantom rate for passenger in fare class v reassigned from p to pd with delay level zeta
    phantom_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

    theta = {
        (y, P.index(p), P.index(pd), z): phantom_rates[z]
        for y in Y
        for p in P
        for pd in P
        for z in Z
    }

    # Capacity of arrival and departure slots
    scA = {asl: 5 for asl in AA}
    scD = {dsl: 5 for dsl in DA}

    # Scheduled buffer time for each flight
    sb = {f: 0 for f in F}

    # minimum turn time between flight f and fd with tail t
    mtt = {(t, f, fd): 1 for t in T for f in F for fd in F}

    # minimum connection time between flight f and fd in itinerary p
    mct = {(P.index(p), f, fd): 1 for p in P for f in F for fd in F}

    # Planned connection time between flights f and fd. It equals scheduled departure time of
    # flight fd minus the scheduled arrival time of flight f.
    ct = {(f, fd): max(0, std[fd] - sta[f]) for fd in F for f in F}

    # set of ordered flight pairs of consecutive flights in itinary p.
    CF_p = {P.index(p): [(p[i], p[i + 1]) for i, _ in enumerate(p[:-1])] for p in P}

    # One if flight f is the last flight of itinerary p, and zero otherwise.
    lf = {
        (P.index(p), f): (lambda last: 1 if last == f else 0)(p[-1])
        for p in P
        for f in F
    }

    # Upper bound on the delay, expressed in hours, corresponding to delay level ζ.
    small_theta = {0: 1, 1: 2, 2: 5, 3: 10, 4: 72}

    # Extra fuel cost for delay absorption (through cruise speed increases) per hour for flight f.
    fc = {f: 30000 for f in F}

    # Sum of the cost of the loss of goodwill and the compensation cost (if any) for a
    # passenger who was scheduled to take itinerary p and is reassigned to itinerary p’, if
    # the passenger’s destination arrival delay via itinerary p′ compared with the planned
    # arrival time of itinerary p corresponds to delay level ζ
    pc = {(z, P.index(p), P.index(pd)): 250 for z in Z for p in P for pd in P}

    # Initial unbounded, first solve will handle.
    kappa = 0
    x_hat = {(f, t): 0 for f in F for t in T}

    # Starting location of planes (binary)
    tb = {(t, k): 0 for t in T for k in K}

    P_sorted = sorted(P, key=len, reverse=True)

    tail_count = 0
    for airport in K:
        deperatures = FD_k[airport]
        for deperature in deperatures:
            for itin in P_sorted:
                if (
                    deperature in itin
                    and itin.index(deperature) == 0
                    and 1 not in [x_hat[(deperature, tail)] for tail in T]
                ):
                    tb[(tail_count, airport)] = 1
                    tail_count += 1

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


def test_standard_solve():
    """
    Generates the base data, runs a solve to get x_hat, and then runs another solve and confirms
    the value is as expected and no rescheduling has to occur.
    """

    standard_solve = Model("test_standard_solve")

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

    variables = generate_variables(standard_solve, T, F, Y, Z, P, AA, DA, CO_p)
    set_objective(
        standard_solve,
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
    flight_scheduling_constraints(standard_solve, variables, F, T_f)
    sequencing_and_fleet_size_constraints(
        standard_solve, variables, T, F, K, F_t, T_f, FD_k, CF_f, tb
    )
    passenger_flow_constraints(
        standard_solve, variables, F, Y, Z, P, T_f, CO_p, theta, n, q
    )
    airport_slot_constraints(
        standard_solve, variables, F, Z, sta, std, AA, DA, AAF, DAF, FAA, FDA, scA, scD
    )
    flight_delay_constraints(standard_solve, variables, T, F, T_f, CF_f, sb, mtt, ct)
    itinerary_feasibility_constraints(
        standard_solve, variables, F, P, sta, std, CF_p, mct
    )
    itinerary_delay_constraints(
        standard_solve, variables, F, Z, P, sta, CO_p, lf, small_theta
    )
    beta_linearizing_constraints(standard_solve, variables, Y, Z, P, CO_p)

    print("optimizing to get xhat...")
    standard_solve.setParam("OutputFlag", 0)
    standard_solve.optimize()

    original_obj_val = standard_solve.objVal

    x_hat = generate_x_hat(standard_solve, variables, F, T)
    kappa = 1000

    set_objective(
        standard_solve,
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

    print("optimizing...")
    standard_solve.setParam("OutputFlag", 1)
    standard_solve.optimize()

    print("generating output...")
    generate_output(
        standard_solve,
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
    )

    assert round(standard_solve.objVal, 2) == round(original_obj_val, 2)


def test_reschedule_slot_cancel():
    """
    Generates the base data, runs a solve to get x_hat, and then removes the arrival slot
    for flight 0, causing it to be delayed to the next slot. This causes the objective value to
    increase and some itineraries to be disrupted.
    """

    test_reschedule_slot_cancel = Model("test_reschedule_slot_cancel")

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

    variables = generate_variables(
        test_reschedule_slot_cancel, T, F, Y, Z, P, AA, DA, CO_p
    )
    set_objective(
        test_reschedule_slot_cancel,
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
    flight_scheduling_constraints(test_reschedule_slot_cancel, variables, F, T_f)
    sequencing_and_fleet_size_constraints(
        test_reschedule_slot_cancel, variables, T, F, K, F_t, T_f, FD_k, CF_f, tb
    )
    passenger_flow_constraints(
        test_reschedule_slot_cancel, variables, F, Y, Z, P, T_f, CO_p, theta, n, q
    )
    airport_slot_constraints(
        test_reschedule_slot_cancel,
        variables,
        F,
        Z,
        sta,
        std,
        AA,
        DA,
        AAF,
        DAF,
        FAA,
        FDA,
        scA,
        scD,
    )
    flight_delay_constraints(
        test_reschedule_slot_cancel, variables, T, F, T_f, CF_f, sb, mtt, ct
    )
    itinerary_feasibility_constraints(
        test_reschedule_slot_cancel, variables, F, P, sta, std, CF_p, mct
    )
    itinerary_delay_constraints(
        test_reschedule_slot_cancel, variables, F, Z, P, sta, CO_p, lf, small_theta
    )
    beta_linearizing_constraints(test_reschedule_slot_cancel, variables, Y, Z, P, CO_p)

    print("optimizing to get xhat...")
    test_reschedule_slot_cancel.setParam("OutputFlag", 0)
    test_reschedule_slot_cancel.optimize()

    x_hat = generate_x_hat(test_reschedule_slot_cancel, variables, F, T)

    # Delay flight 0 by makings its arrival slot unavailable.
    AA.remove((54.0, 56.0))
    kappa = 1000

    # Generate new data
    AAF = {f: [i for i, slot in enumerate(AA) if sta[f] <= slot[0]] for f in F}
    DAF = {f: [i for i, slot in enumerate(DA) if std[f] <= slot[0]] for f in F}

    FAA = {asl: [f for f in F if sta[f] <= asl[0]] for asl in AA}
    FDA = {dsl: [f for f in F if std[f] <= dsl[0]] for dsl in DA}

    print("Regenerate neccecary constraints...")
    airport_slot_constraints(
        test_reschedule_slot_cancel,
        variables,
        F,
        Z,
        sta,
        std,
        AA,
        DA,
        AAF,
        DAF,
        FAA,
        FDA,
        scA,
        scD,
    )
    itinerary_feasibility_constraints(
        test_reschedule_slot_cancel, variables, F, P, sta, std, CF_p, mct
    )
    itinerary_delay_constraints(
        test_reschedule_slot_cancel, variables, F, Z, P, sta, CO_p, lf, small_theta
    )
    set_objective(
        test_reschedule_slot_cancel,
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

    print("optimizing...")
    test_reschedule_slot_cancel.setParam("OutputFlag", 1)
    test_reschedule_slot_cancel.optimize()

    print("generating output...")
    generate_output(
        test_reschedule_slot_cancel,
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
    )

    assert round(test_reschedule_slot_cancel.objVal, 5) == 1313999.99845


def test_reschedule_flight_cancel():
    """
    Generates the base data, runs a solve to get x_hat, and then remove flight 0.
    This causes the objective value to increase and some itineraries to be disrupted.
    """

    test_reschedule_flight_cancel = Model("test_reschedule_flight_cancel")

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

    variables = generate_variables(
        test_reschedule_flight_cancel, T, F, Y, Z, P, AA, DA, CO_p
    )
    set_objective(
        test_reschedule_flight_cancel,
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
    flight_scheduling_constraints(test_reschedule_flight_cancel, variables, F, T_f)
    sequencing_and_fleet_size_constraints(
        test_reschedule_flight_cancel, variables, T, F, K, F_t, T_f, FD_k, CF_f, tb
    )
    passenger_flow_constraints(
        test_reschedule_flight_cancel, variables, F, Y, Z, P, T_f, CO_p, theta, n, q
    )
    airport_slot_constraints(
        test_reschedule_flight_cancel,
        variables,
        F,
        Z,
        sta,
        std,
        AA,
        DA,
        AAF,
        DAF,
        FAA,
        FDA,
        scA,
        scD,
    )
    flight_delay_constraints(
        test_reschedule_flight_cancel, variables, T, F, T_f, CF_f, sb, mtt, ct
    )
    itinerary_feasibility_constraints(
        test_reschedule_flight_cancel, variables, F, P, sta, std, CF_p, mct
    )
    itinerary_delay_constraints(
        test_reschedule_flight_cancel, variables, F, Z, P, sta, CO_p, lf, small_theta
    )
    beta_linearizing_constraints(
        test_reschedule_flight_cancel, variables, Y, Z, P, CO_p
    )

    print("optimizing to get xhat...")
    test_reschedule_flight_cancel.setParam("OutputFlag", 0)
    test_reschedule_flight_cancel.optimize()

    original_obj_val = test_reschedule_flight_cancel.objVal

    x_hat = generate_x_hat(test_reschedule_flight_cancel, variables, F, T)
    kappa = 1000

    print("Regenerate neccecary constraints...")
    airport_slot_constraints(
        test_reschedule_flight_cancel,
        variables,
        F,
        Z,
        sta,
        std,
        AA,
        DA,
        AAF,
        DAF,
        FAA,
        FDA,
        scA,
        scD,
    )
    itinerary_feasibility_constraints(
        test_reschedule_flight_cancel, variables, F, P, sta, std, CF_p, mct
    )
    itinerary_delay_constraints(
        test_reschedule_flight_cancel, variables, F, Z, P, sta, CO_p, lf, small_theta
    )
    set_objective(
        test_reschedule_flight_cancel,
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

    print("optimizing...")
    test_reschedule_flight_cancel.setParam("OutputFlag", 1)

    _, z, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = variables
    z[12].lb = 1
    z[12].ub = 1

    test_reschedule_flight_cancel.optimize()

    print("generating output...")
    generate_output(
        test_reschedule_flight_cancel,
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
    )

    assert round(test_reschedule_flight_cancel.objVal, 5) == 1193000.00149


def test_reschedule_airport_shutdown():
    """
    Generates the base data, runs a solve to get x_hat, and then closes down the BNE airport
    between hours 50 & 55.

    Because the airport closures are typically out of control of an individual airline,
    Regulation (EC) No 261/2004 does not apply, and passengers are not required to be compensated.
    However, they must be provided with reaccommodation.
    """

    test_reschedule_airport_shutdown = Model("test_reschedule_airport_shutdown")

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

    variables = generate_variables(
        test_reschedule_airport_shutdown, T, F, Y, Z, P, AA, DA, CO_p
    )
    set_objective(
        test_reschedule_airport_shutdown,
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
    flight_scheduling_constraints(test_reschedule_airport_shutdown, variables, F, T_f)
    sequencing_and_fleet_size_constraints(
        test_reschedule_airport_shutdown, variables, T, F, K, F_t, T_f, FD_k, CF_f, tb
    )
    passenger_flow_constraints(
        test_reschedule_airport_shutdown, variables, F, Y, Z, P, T_f, CO_p, theta, n, q
    )
    airport_slot_constraints(
        test_reschedule_airport_shutdown,
        variables,
        F,
        Z,
        sta,
        std,
        AA,
        DA,
        AAF,
        DAF,
        FAA,
        FDA,
        scA,
        scD,
    )
    flight_delay_constraints(
        test_reschedule_airport_shutdown, variables, T, F, T_f, CF_f, sb, mtt, ct
    )
    itinerary_feasibility_constraints(
        test_reschedule_airport_shutdown, variables, F, P, sta, std, CF_p, mct
    )
    itinerary_delay_constraints(
        test_reschedule_airport_shutdown, variables, F, Z, P, sta, CO_p, lf, small_theta
    )
    beta_linearizing_constraints(
        test_reschedule_airport_shutdown, variables, Y, Z, P, CO_p
    )

    print("optimizing to get xhat...")
    test_reschedule_airport_shutdown.setParam("OutputFlag", 0)
    test_reschedule_airport_shutdown.optimize()

    original_obj_val = test_reschedule_airport_shutdown.objVal

    # Close down airport 'BNE' between hours 50 & 55:
    _, _, _, _, _, _, _, _, _, deltaA, deltaD, _, _, _, _, _ = variables

    for node in graph.adj_list.keys():
        for neigh in graph.get_neighbours(node):
            if neigh[1] is not None:
                # Departing Brisbane
                if all(
                    [
                        node.get_name() == "SYD",
                        node.time >= 50,
                        node.time <= 70,
                    ]
                ):
                    deltaA[neigh[1]].lb = 70 - sta[neigh[1]]
                    deltaD[neigh[1]].lb = 70 - std[neigh[1]]

                # Arriving to Brisbane
                if all(
                    [
                        neigh[0].get_name() == "SYD",
                        neigh[0].time >= 50,
                        neigh[0].time <= 70,
                    ]
                ):
                    deltaA[neigh[1]].lb = 70 - sta[neigh[1]]
                    deltaD[neigh[1]].lb = 70 - std[neigh[1]]

    x_hat = generate_x_hat(test_reschedule_airport_shutdown, variables, F, T)
    kappa = 1000

    print("Regenerate neccecary constraints...")
    pc = {(z, P.index(p), P.index(pd)): 0 for z in Z for p in P for pd in P}
    set_objective(
        test_reschedule_airport_shutdown,
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

    print("optimizing...")
    test_reschedule_airport_shutdown.setParam("OutputFlag", 1)
    test_reschedule_airport_shutdown.optimize()

    print("generating output...")
    generate_output(
        test_reschedule_airport_shutdown,
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
    )

    assert round(test_reschedule_airport_shutdown.objVal, 6) == 1877999.968588
