from gurobipy import *
from airline_recovery import *
from data.build_psuedo_aus import *
import random
import numpy as np
from math import floor

random.seed(59)
TIME_HORIZON = 72

num_flights = floor(random.normalvariate(20, 1))
flight_distribution = divide_number(num_flights, len(AIRPORTS), 0.25, 0.35)
graph = create_graph(flight_distribution)
print("graph created")

itin_classes = {1: 20, 2: 5, 3: 1}

try:
    P = generate_itineraries(graph, itin_classes, [])
except RecursionError:
    print("ERROR: Recursion depth exceeded, please reduce itinerary length")

P.insert(0, [])
print("\nitineraries created")
print(P, "\n")


print("Graph")
for node, neigh in graph.adj_list.items():
    if len([n for n in neigh if n[1] is not None]) > 0:
        print(node, ": ", [n for n in neigh if n[1] is not None])
print()


def build_base_data() -> tuple:
    num_flights = graph.count_all_flights()
    num_tails = 20
    num_airports = 10
    num_fare_classes = 2
    num_delay_levels = 5
    num_time_instances = 8

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
        ]
        for p in P
        if p != []
    }

    # Manually define the the compatibility of the empty itinerary
    CO_p[0] = [0]

    # Manually add the empty itinerary as a compatible itinerary for each itinerary
    for p in P:
        CO_p[P.index(p)].append(0)

    # Cost of operating flight f with tail t
    oc = {(t, f): 10000 for t in T for f in F}

    # Delay cost per hour of arrival delay of flight f
    dc = {f: 12500 for f in F}

    # Number of passengers in fare class v that are originally scheduled to take itinerary p
    n = {(v, P.index(p)): 50 for v in Y for p in P}

    # Seating capacity of tail t in T
    q = {t: 250 for t in T}

    # Reaccommodation Cost for a passenger reassigned from p to pd.
    rc_costs = {time: 100 for time in range(0, 4)}
    for time in range(3, 7):
        rc_costs[time] = 400
    for time in range(6, 17):
        rc_costs[time] = 600
    for time in range(16, 72):
        rc_costs[time] = 1000

    rc = {}
    for p in P:
        for pd in P:
            if p != [] and pd != [] and std[pd[0]] >= std[p[0]]:
                time_diff = floor(std[pd[0]] - std[p[0]])
                rc[(P.index(p), P.index(pd))] = rc_costs[time_diff]

    for p in P:
        rc[(P.index(p), 0)] = 1600

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
        (P.index(p), f): (lambda last: 1 if last == f else 0)(p[-1] if p != [] else 0)
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
    pc = {(z, P.index(p), P.index(pd)): 200 + z * 150 for z in Z for p in P for pd in P}

    # Initial unbounded, first solve will handle.
    kappa = 0
    x_hat = {(f, t): 0 for f in F for t in T}

    # Starting location of planes (binary)
    tb = {(t, k): 0 for t in T for k in K}

    P_sorted = sorted(P, key=len, reverse=True)

    tail_count = 0
    for itin in P_sorted:
        if itin:
            airport = DK_f[itin[0]]
            tb[(tail_count, airport)] = 1
            tail_count += 1
            if tail_count == num_tails:
                break

    # Create Maintenance data / sets
    PI = range(num_time_instances)
    MO = set(
        [
            (f, fd)
            for f in F
            for fd in F
            if std[f] < std[fd]
            and AK_f[f] == DK_f[fd]
            and (AK_f[f] == "SYD" or AK_f[f] == "BNE")
        ]
    )
    F_pi = {
        pi: [f for f in F if sta[f] <= (1 + pi) * (TIME_HORIZON / num_time_instances)]
        for pi in PI
    }
    K_m = {"SYD", "BNE"}
    T_m = set()
    PI_m = set(PI)

    abh = {t: TIME_HORIZON for t in T}
    sbh = {f: TIME_HORIZON for f in F}

    mbh = {t: 14 * 24 for t in T}
    mt = {t: 12 for t in T}
    aw = {k: (lambda k: 5 if k == "SYD" or k == "BNE" else 0)(k) for k in K}

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
        PI,
        MO,
        F_pi,
        K_m,
        T_m,
        PI_m,
        abh,
        sbh,
        mbh,
        mt,
        aw,
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
        PI,
        MO,
        F_pi,
        K_m,
        T_m,
        PI_m,
        abh,
        sbh,
        mbh,
        mt,
        aw,
    ) = build_base_data()

    variables = generate_variables(standard_solve, T, F, Y, Z, P, AA, DA, CO_p, K)
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
        T_m,
    )

    (
        _,
        z,
        _,
        _,
        _,
        _,
        _,
        lambd,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = variables

    # Obj the same, correct itins are disrupted and flights cancelled
    assert round(standard_solve.objVal, 2) == round(original_obj_val, 2)
    for f in F:
        if f == 12 or f == 14 or f == 17:
            assert z[f].x > 0.9
        else:
            assert z[f].x < 0.9
    for p in P:
        if P.index(p) == 5 or P.index(p) == 16 or P.index(p) == 18:
            assert lambd[P.index(p)].x > 0.9
        else:
            assert lambd[P.index(p)].x < 0.9


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
        PI,
        MO,
        F_pi,
        K_m,
        T_m,
        PI_m,
        abh,
        sbh,
        mbh,
        mt,
        aw,
    ) = build_base_data()

    variables = generate_variables(
        test_reschedule_slot_cancel, T, F, Y, Z, P, AA, DA, CO_p, K
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
        T_m,
    )

    (
        _,
        z,
        _,
        _,
        _,
        _,
        h,
        lambd,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = variables

    # Cancellations and reschedules occur as expected
    for f in F:
        if f == 6 or f == 10 or f == 12 or f == 14 or f == 17:
            assert z[f].x > 0.9
        else:
            assert z[f].x < 0.9
    for p in P:
        if (
            P.index(p) == 3
            or P.index(p) == 7
            or P.index(p) == 5
            or P.index(p) == 16
            or P.index(p) == 18
        ):
            assert lambd[P.index(p)].x > 0.9
        else:
            assert lambd[P.index(p)].x < 0.9
    for p in P:
        for pd in P:
            if p != pd:
                for v in Y:
                    if any(
                        [
                            P.index(p) == 3 and P.index(pd) == 2,
                            P.index(p) == 7 and P.index(pd) == 1,
                            P.index(p) == 5 and P.index(pd) == 0,
                            P.index(p) == 16 and P.index(pd) == 15,
                            P.index(p) == 18 and P.index(pd) == 0,
                        ]
                    ):
                        assert int(h[P.index(p), P.index(pd), v].x) == 50
                    else:
                        assert int(h[P.index(p), P.index(pd), v].x) == 0


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
        PI,
        MO,
        F_pi,
        K_m,
        T_m,
        PI_m,
        abh,
        sbh,
        mbh,
        mt,
        aw,
    ) = build_base_data()

    variables = generate_variables(
        test_reschedule_flight_cancel, T, F, Y, Z, P, AA, DA, CO_p, K
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

    (
        _,
        z,
        _,
        _,
        _,
        _,
        h,
        lambd,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = variables

    z[2].lb = 1
    z[2].ub = 1

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
        T_m,
    )

    (
        _,
        z,
        _,
        _,
        _,
        _,
        h,
        lambd,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = variables

    for f in F:
        if f == 12 or f == 14 or f == 17 or f == 2:
            assert z[f].x > 0.9
        else:
            assert z[f].x < 0.9
    for p in P:
        if P.index(p) == 16 or P.index(p) == 18 or P.index(p) == 5 or P.index(p) == 12:
            assert lambd[P.index(p)].x > 0.9
        else:
            assert lambd[P.index(p)].x < 0.9
    for p in P:
        for pd in P:
            if p != pd:
                for v in Y:
                    if P.index(p) == 5 and P.index(pd) == 0:
                        assert int(h[P.index(p), P.index(pd), v].x) == 50
                    elif P.index(p) == 12 and P.index(pd) == 21:
                        assert int(h[P.index(p), P.index(pd), v].x) == 50
                    elif P.index(p) == 18 and P.index(pd) == 0:
                        assert int(h[P.index(p), P.index(pd), v].x) == 50
                    elif P.index(p) == 16 and P.index(pd) == 15:
                        assert int(h[P.index(p), P.index(pd), v].x) == 50
                    elif P.index(p) == 26 and P.index(pd) == 0:
                        assert (int(h[P.index(p), P.index(pd), 1].x) == 40) or (
                            int(h[P.index(p), P.index(pd), 0].x) == 40
                        )
                    else:
                        assert int(h[P.index(p), P.index(pd), v].x) == 0


def test_reschedule_airport_shutdown():
    """
    Generates the base data, runs a solve to get x_hat, and then closes down the SYD airport
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
        PI,
        MO,
        F_pi,
        K_m,
        T_m,
        PI_m,
        abh,
        sbh,
        mbh,
        mt,
        aw,
    ) = build_base_data()

    variables = generate_variables(
        test_reschedule_airport_shutdown, T, F, Y, Z, P, AA, DA, CO_p, K
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

    # Close down airport 'SYD' between hours 50 & 70:
    (
        _,
        z,
        _,
        _,
        _,
        _,
        h,
        lambd,
        _,
        deltaA,
        deltaD,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = variables

    for node in graph.adj_list.keys():
        for neigh, flightid in graph.get_neighbours(node):
            if flightid is not None:
                # Departing Sydney
                if all(
                    [
                        node.get_name() == "SYD",
                        node.time >= 50,
                        node.time <= 70,
                    ]
                ):
                    deltaA[flightid].lb = 70 - sta[flightid]
                    deltaD[flightid].lb = 70 - std[flightid]

                # Arriving to Sydney
                if all(
                    [
                        neigh.get_name() == "SYD",
                        neigh.time >= 50,
                        neigh.time <= 70,
                    ]
                ):
                    deltaA[flightid].lb = 70 - sta[flightid]
                    deltaD[flightid].lb = 70 - std[flightid]

    x_hat = generate_x_hat(test_reschedule_airport_shutdown, variables, F, T)
    kappa = 1000

    print("Regenerate neccecary constraints...")
    for p in P:
        for pd in P:
            for z in Z:
                for f in p:
                    if (std[f] >= 50 and std[f] <= 70) and (DK_f[f] == "SYD"):
                        pc[(z, P.index(p), P.index(pd))] = 0

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
        T_m,
    )

    (
        _,
        z,
        _,
        _,
        _,
        _,
        h,
        lambd,
        _,
        deltaA,
        deltaD,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = variables

    for f in F:
        if f == 0 or f == 6 or f == 12 or f == 14 or f == 17:
            assert z[f].x > 0.9
        else:
            assert z[f].x < 0.9
    for p in P:
        if (
            P.index(p) == 3
            or P.index(p) == 5
            or P.index(p) == 11
            or P.index(p) == 16
            or P.index(p) == 18
            or P.index(p) == 23
            or P.index(p) == 21
        ):
            assert lambd[P.index(p)].x > 0.9
        else:
            assert lambd[P.index(p)].x < 0.9
    for p in P:
        for pd in P:
            if p != pd:
                for v in Y:
                    if any(
                        [
                            P.index(p) == 3 and P.index(pd) == 2,
                            P.index(p) == 5 and P.index(pd) == 0,
                            P.index(p) == 11 and P.index(pd) == 2,
                            P.index(p) == 16 and P.index(pd) == 15,
                            P.index(p) == 18 and P.index(pd) == 0,
                            P.index(p) == 21 and P.index(pd) == 0,
                            P.index(p) == 23 and P.index(pd) == 0,
                        ]
                    ):
                        assert int(h[P.index(p), P.index(pd), v].x) == 50
                    else:
                        assert int(h[P.index(p), P.index(pd), v].x) == 0


def test_maintenance():
    """
    Generates the base data, runs a solve to get x_hat, and then runs another solve and confirms
    the value is as expected and no rescheduling has to occur.
    """

    standard_solve_with_maintenance = Model("test_standard_solve_with_maintenance")

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
        PI,
        MO,
        F_pi,
        K_m,
        T_m,
        PI_m,
        abh,
        sbh,
        mbh,
        mt,
        aw,
    ) = build_base_data()

    variables = generate_variables(
        standard_solve_with_maintenance, T, F, Y, Z, P, AA, DA, CO_p, K
    )
    set_objective(
        standard_solve_with_maintenance,
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
    flight_scheduling_constraints(standard_solve_with_maintenance, variables, F, T_f)
    sequencing_and_fleet_size_constraints(
        standard_solve_with_maintenance, variables, T, F, K, F_t, T_f, FD_k, CF_f, tb
    )
    passenger_flow_constraints(
        standard_solve_with_maintenance, variables, F, Y, Z, P, T_f, CO_p, theta, n, q
    )
    airport_slot_constraints(
        standard_solve_with_maintenance,
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
        standard_solve_with_maintenance, variables, T, F, T_f, CF_f, sb, mtt, ct
    )
    itinerary_feasibility_constraints(
        standard_solve_with_maintenance, variables, F, P, sta, std, CF_p, mct
    )
    itinerary_delay_constraints(
        standard_solve_with_maintenance, variables, F, Z, P, sta, CO_p, lf, small_theta
    )
    maintenance_schedule_constraints(
        standard_solve_with_maintenance, variables, T_m, sta, T_f, F, F_t, mt, MO, std
    )
    workshop_schedule_constraints(
        standard_solve_with_maintenance, variables, F_t, T_m, K_m, F, T_f, K, aw, FA_k
    )
    maintenance_check_constraints(
        standard_solve_with_maintenance,
        variables,
        T_m,
        PI_m,
        F_pi,
        sbh,
        mbh,
        F_t,
        MO,
        abh,
        F,
        T_f,
    )
    beta_linearizing_constraints(
        standard_solve_with_maintenance, variables, Y, Z, P, CO_p
    )

    print("optimizing to get xhat...")
    standard_solve_with_maintenance.setParam("OutputFlag", 0)
    standard_solve_with_maintenance.optimize()

    original_obj_val = standard_solve_with_maintenance.objVal

    x_hat = generate_x_hat(standard_solve_with_maintenance, variables, F, T)
    kappa = 1000

    # Include tails that need maintenance for this test
    T_m = {3, 10}

    # get the time of the first flight that uses tail t
    abh = {t: 0 for t in T}

    # For each flight, find the time of the next flight that uses the same tail
    next_f = {f: TIME_HORIZON for f in F}
    for f, t in x_hat.keys():
        if x_hat[(f, t)] == 1:
            for fd in F:
                if fd != f and x_hat[(fd, t)] == 1 and std[fd] > sta[f]:
                    if std[fd] < next_f[f]:
                        next_f[f] = std[fd]

    sbh = {f: next_f[f] - sta[f] for f in F}

    standard_solve_with_maintenance_2nd_run = Model(
        "test_standard_solve_with_maintenance_2nd_run"
    )

    variables = generate_variables(
        standard_solve_with_maintenance_2nd_run, T, F, Y, Z, P, AA, DA, CO_p, K
    )
    set_objective(
        standard_solve_with_maintenance_2nd_run,
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
    flight_scheduling_constraints(
        standard_solve_with_maintenance_2nd_run, variables, F, T_f
    )
    sequencing_and_fleet_size_constraints(
        standard_solve_with_maintenance_2nd_run,
        variables,
        T,
        F,
        K,
        F_t,
        T_f,
        FD_k,
        CF_f,
        tb,
    )
    passenger_flow_constraints(
        standard_solve_with_maintenance_2nd_run,
        variables,
        F,
        Y,
        Z,
        P,
        T_f,
        CO_p,
        theta,
        n,
        q,
    )
    airport_slot_constraints(
        standard_solve_with_maintenance_2nd_run,
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
        standard_solve_with_maintenance_2nd_run, variables, T, F, T_f, CF_f, sb, mtt, ct
    )
    itinerary_feasibility_constraints(
        standard_solve_with_maintenance_2nd_run, variables, F, P, sta, std, CF_p, mct
    )
    itinerary_delay_constraints(
        standard_solve_with_maintenance_2nd_run,
        variables,
        F,
        Z,
        P,
        sta,
        CO_p,
        lf,
        small_theta,
    )
    maintenance_schedule_constraints(
        standard_solve_with_maintenance_2nd_run,
        variables,
        T_m,
        sta,
        T_f,
        F,
        F_t,
        mt,
        MO,
        std,
    )
    workshop_schedule_constraints(
        standard_solve_with_maintenance_2nd_run,
        variables,
        F_t,
        T_m,
        K_m,
        F,
        T_f,
        K,
        aw,
        FA_k,
    )
    maintenance_check_constraints(
        standard_solve_with_maintenance_2nd_run,
        variables,
        T_m,
        PI_m,
        F_pi,
        sbh,
        mbh,
        F_t,
        MO,
        abh,
        F,
        T_f,
    )
    beta_linearizing_constraints(
        standard_solve_with_maintenance_2nd_run, variables, Y, Z, P, CO_p
    )

    print("optimizing...")
    standard_solve_with_maintenance_2nd_run.setParam("OutputFlag", 1)
    standard_solve_with_maintenance_2nd_run.optimize()

    print("generating output...")
    generate_output(
        standard_solve_with_maintenance_2nd_run,
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

    # Maintenance occurs on the correct tails in the correct time window
    assert round(imt[10].x, 2) == 38 and round(fmt[10].x, 2) == 50
    assert round(imt[3].x, 2) == 42.4 and round(fmt[3].x, 2) == 54.4
