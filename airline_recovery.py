from gurobipy import *

BIG_M = 999999999

# QUESTIONS:
# is rho redundant?
# is the inequality of constraint (6) backwards?
# Can tb just be a list of the starting location? Can a plane be in two places at once?


def airline_recovery_basic() -> None:
    """
    basic initial implementation of Passenger-Centric Integrated Airline Schedule and
    Aircraft Recovery.

    Initial Data Setup:

    - Both planes have 100 passengers (and a capacity of 100 passengers)
    - There is only 1 fare class
    - only one delay level (shouldn't be any delays)
    - Operating cost of flights = 1
    - Phantom rate currently set to 0
    - Each capacity and arrival slot has capacity of 2

    - Currently no rescheduling has to occur

    ITINERARIES
    T0: F0 (depart A0, arrive A1) -> F1 (depart A1, arrive A2)
    T1: F2 (depart A0, arrive A1) -> F3 (depart A1, arrive A2)

    The scheduled arrival/departure of these planes can be seen below
    """

    num_flights = 4
    num_tails = 2
    num_airports = 3
    num_fare_classes = 1
    num_delay_levels = 1

    # Sets
    T = range(num_tails)
    F = range(num_flights)
    P = [[0, 1], [2, 3]]  # Set of itineraries
    K = range(num_airports)
    Y = range(num_fare_classes)
    Z = range(num_delay_levels)

    # Scheduled arrival (departure) time for flight f in F
    std = [1, 2, 9.5, 10.5]
    sta = [6, 7, 13, 13.5]

    # Arrival and Depature slots
    AA = [[5, 7], [6, 8], [12, 14], [13, 14]]
    DA = [[0, 2], [1, 3], [9, 10], [10, 11]]

    # Set of arrival and departure slots compatible with flight f
    AAF = [[0, 1], [0, 1], [2, 3], [2, 3]]
    DAF = [[0, 1], [0, 1], [2], [3]]

    # Set of flights compatible with arrive/departure slot asl/dsl
    FAA = [[0, 1], [0, 1], [2, 3], [2, 3]]
    FDA = [[0, 1], [0, 1], [2], [3]]

    # set of flights compatible with tail T
    F_t = [[0, 1], [2, 3]]

    # set of tails compatible with flight F
    T_f = [[0], [0], [1], [1]]

    # Set of flights compatible with a connection from flight f
    CF_f = [[1], [], [3], []]

    # Set of flights f which arrive to airport k
    FA_k = [[], [0, 2], [1, 3]]

    # Set of flights f which depart from airport k
    FD_k = [[0, 2], [1, 3], []]

    # Subset of itineraries compatible with a reasignment from an original itinerary p.
    CO_p = [[0], [1]]

    # Data

    # Cost of operating flight f with tail t
    oc = [[1 for _ in range(num_flights)] for _ in range(num_tails)]

    # Delay cost per minute of arrival delay of flight f
    dc = [100 for _ in range(num_flights)]

    # Number of passengers in fare class v that are originally scheduled to
    # take itinerary p
    n = [[100, 100]]

    # Seating capacity of tail t in T
    q = [100, 100]

    # Reaccommodation Cost for a passenger reassigned from p to pd.
    rc = [[(lambda p, pd: 0 if p == pd else 0.5)(p, pd) for p in P] for pd in P]

    # Phantom rate for passenger in fare class v reassigned from p to pd with delay level
    # zeta
    theta = [
        [[[0 for _ in Y] for _ in range(len(P))] for _ in range(len(P))] for _ in Z
    ]

    # Starting location of planes (binary)
    tb = [[1, 0, 0], [1, 0, 0]]

    # Capacity of arrival and departure slots
    scA = [1, 1, 1, 1]
    scD = [1, 1, 1, 1]

    m = Model("airline recovery basic")

    # Variables
    x = {(t, f): m.addVar(vtype=GRB.BINARY) for t in T for f in F}
    z = {f: m.addVar(vtype=GRB.BINARY) for f in F}
    y = {(f, fd): m.addVar(vtype=GRB.BINARY) for f in F for fd in F if fd != f}
    sigma = {f: m.addVar(vtype=GRB.BINARY) for f in F}
    rho = {f: m.addVar(vtype=GRB.BINARY) for f in F}
    phi = {(t, f): m.addVar(vtype=GRB.BINARY) for f in F for t in T}
    h = {
        (p, pd, v): m.addVar() for v in Y for p in range(len(P)) for pd in range(len(P))
    }
    lambd = {p: m.addVar(vtype=GRB.BINARY) for p in range(len(P))}
    alpha = {
        (p, pd, zeta): m.addVar(vtype=GRB.BINARY)
        for p in range(len(P))
        for pd in range(len(P))
        for zeta in Z
    }
    deltaA = {f: m.addVar() for f in F}
    deltaD = {f: m.addVar() for f in F}
    vA = {(asl, f): m.addVar(vtype=GRB.BINARY) for f in F for asl in range(len(AA))}
    vD = {(dsl, f): m.addVar(vtype=GRB.BINARY) for f in F for dsl in range(len(DA))}

    # Objective (currently just operating cost and reassignment cost)
    m.setObjective(
        (
            quicksum(oc[t][f] * x[t, f] for t in T for f in F_t[t])
            + quicksum(
                rc[p][pd]
                * (
                    1
                    - quicksum(theta[v][p][pd][zeta] * alpha[p, pd, zeta] for zeta in Z)
                )
                * h[p, pd, v]
                for v in Y
                for p in range(len(P))
                for pd in CO_p[p]
            )
        ),
        GRB.MINIMIZE,
    )

    # Flight Scheduling Constraints
    fsc_1 = {f: m.addConstr(quicksum(x[t, f] for t in T_f[f]) + z[f] == 1) for f in F}

    # Sequencing & Fleet Size Constraints
    sfsc_1 = {
        f: m.addConstr(quicksum(y[f, fd] for fd in CF_f[f]) + sigma[f] == 1 - z[f])
        for f in F
    }

    sfsc_2 = {
        fd: m.addConstr(
            quicksum(y[f, fd] for f in F if fd in CF_f[f]) + rho[fd] == 1 - z[fd]
        )
        for fd in F
    }

    sfsc_3 = {
        (f, fd, t): m.addConstr(1 + x[t, fd] >= x[t, f] + y[f, fd])
        for f in F
        for fd in CF_f[f]
        for t in list(set(T_f[f]).intersection(T_f[fd]))
    }

    sfsc_4 = {
        (f, fd, t): m.addConstr(1 + x[t, f] <= x[t, fd] + y[f, fd])
        for f in F
        for fd in CF_f[f]
        for t in list(set(T_f[f]).intersection(T_f[fd]))
    }

    sfsc_5 = {
        (t, f): m.addConstr(rho[f] + x[t, f] <= 1 + phi[t, f])
        for f in F
        for t in T_f[f]
    }

    sfsc_6 = {
        (t, k): m.addConstr(
            quicksum(phi[t, f] for f in list(set(F_t[t]).intersection(FD_k[k])))
            <= tb[t][k]
        )
        for t in T
        for k in K
    }

    # Passenger Flow Constraints
    pfc_1 = {
        (p, v): m.addConstr(quicksum(h[p, pd, v] for pd in CO_p[p]) == n[v][p])
        for p in range(len(P))
        for v in Y
    }

    pfc_2 = {
        (p, v, pd): m.addConstr(h[p, pd, v] <= (1 - lambd[pd]) * n[v][p])
        for p in range(len(P))
        for v in Y
        for pd in CO_p[p]
    }

    pfc_3 = {
        f: m.addConstr(
            quicksum(q[t] * x[t, f] for t in T_f[f])
            >= quicksum(
                (1 - quicksum(theta[v][p][pd][zeta] * alpha[p, pd, zeta] for zeta in Z))
                * h[p, pd, v]
                for v in Y
                for p in range(len(P))
                for pd in CO_p[p]
                if f in P[pd]
            )
        )
        for f in F
    }

    # Airport slot constraints
    asc_1 = {
        (f, asl): m.addConstr(
            AA[asl][0] <= sta[f] + deltaA[f] + BIG_M * (1 - vA[asl, f])
        )
        for f in F
        for asl in AAF[f]
    }

    asc_2 = {
        (f, asl): m.addConstr(
            AA[asl][1] >= sta[f] + deltaA[f] - BIG_M * (1 - vA[asl, f])
        )
        for f in F
        for asl in AAF[f]
    }

    asc_3 = {
        f: m.addConstr(quicksum(vA[asl, f] for asl in AAF[f]) == 1 - z[f]) for f in F
    }

    asc_4 = {
        asl: m.addConstr(quicksum(vA[asl, f] for f in FAA[asl]) <= scA[asl])
        for asl in range(len(AA))
    }

    asc_5 = {
        (f, dsl): m.addConstr(
            DA[dsl][0] <= std[f] + deltaD[f] + BIG_M * (1 - vD[dsl, f])
        )
        for f in F
        for dsl in DAF[f]
    }

    asc_6 = {
        (f, dsl): m.addConstr(
            DA[dsl][1] >= std[f] + deltaD[f] - BIG_M * (1 - vD[dsl, f])
        )
        for f in F
        for dsl in DAF[f]
    }

    asc_7 = {
        f: m.addConstr(quicksum(vD[dsl, f] for dsl in DAF[f]) == 1 - z[f]) for f in F
    }

    asc_8 = {
        dsl: m.addConstr(quicksum(vD[dsl, f] for f in FDA[dsl]) <= scD[dsl])
        for dsl in range(len(DA))
    }

    m.optimize()

    # Generate an output:
    print("\nList of tails which are assigned to flights:")
    for t in T:
        for f in F:
            if x[t, f].x > 0.9:
                print(f"Tail {t} assigned to flight {f}.")

    print("\nList of cancelled flights")
    for f in F:
        if z[f].x > 0.9:
            print(f"Flight {f} has been cancelled.")

    print("\nList of assigned arrival slots:")
    for f in F:
        for asl in range(len(AA)):
            if vA[asl, f].x > 0.9:
                print(f"Flight {f} assigned to arrival slot {AA[asl]}")

    print("\nList of assigned departure slots:")
    for f in F:
        for dsl in range(len(DA)):
            if vD[dsl, f].x > 0.9:
                print(f"Flight {f} assigned to departure slot {DA[dsl]}")

    print("\nDisrupted Itineraries:")
    for p in range(len(P)):
        if lambd[p].x > 0.9:
            print(f"Itinerary {p} is disrupted.")

    print("\nConsecutive Flights:")
    for f in F:
        for fd in F:
            if fd != f:
                if y[f, fd].x > 0.9:
                    print(f"flight {f} is flown and then flight {fd} is flown.")

    print("\nLast Flights:")
    for f in F:
        if sigma[f].x > 0.9:
            print(f"Flight {f} is the last flight.")

    print("\nFirst Flights")
    for f in F:
        for t in T:
            if phi[t, f].x > 0.9:
                print(f"Flight {f} with tail {t} is the first flight.")

    return


if __name__ == "__main__":
    airline_recovery_basic()
