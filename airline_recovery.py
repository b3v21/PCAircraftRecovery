from gurobipy import *
from data.testdata2 import *

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
    gamma = {f: m.addVar() for f in F}

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

    # Every flight must be either flown using exactly one aircraft or must be cancelled
    fsc_1 = {f: m.addConstr(quicksum(x[t, f] for t in T_f[f]) + z[f] == 1) for f in F}

    # Sequencing & Fleet Size Constraints

    # Each non-cancelled flight has another flight after it operated by the same tail,
    # unless it is the last flight in the recovery period operated by that tail
    sfsc_1 = {
        f: m.addConstr(quicksum(y[f, fd] for fd in CF_f[f]) + sigma[f] == 1 - z[f])
        for f in F
    }

    # Each non-cancelled flight has another flight before it operated by the same tail,
    # unless it is the first flight in the recovery period operated by that tail
    sfsc_2 = {
        fd: m.addConstr(
            quicksum(y[f, fd] for f in F if fd in CF_f[f]) + rho[fd] == 1 - z[fd]
        )
        for fd in F
    }

    # Consecutive flights operated by the same tail are assigned the same tail

    # Can't assign tail t to flight f and fly f & fd consecutively by tail t without
    # assigning tail t to flight fd
    sfsc_3 = {
        (f, fd, t): m.addConstr(1 + x[t, fd] >= x[t, f] + y[f, fd])
        for f in F
        for fd in CF_f[f]
        for t in list(set(T_f[f]).intersection(T_f[fd]))
    }

    sfsc_4 = {
        (f, fd, t): m.addConstr(1 + x[t, f] >= x[t, fd] + y[f, fd])
        for f in F
        for fd in CF_f[f]
        for t in list(set(T_f[f]).intersection(T_f[fd]))
    }

    # If a flight is assigned a tail and it is the first flight in a sequence of flights,
    # it must be the first flight of that tail
    sfsc_5 = {
        (t, f): m.addConstr(rho[f] + x[t, f] <= 1 + phi[t, f])
        for f in F
        for t in T_f[f]
    }

    # A flight chosen to be the first flight in a sequence of flights can only be assigned
    # to tails whose initial location matches the flight's departure airport
    sfsc_6 = {
        (t, k): m.addConstr(
            quicksum(phi[t, f] for f in list(set(F_t[t]).intersection(FD_k[k])))
            <= tb[t][k]
        )
        for t in T
        for k in K
    }

    # NOTE: Sequencing & Fleet Size Constraints still allow for a tail to not be used
    # at all during the recovery period

    # Passenger Flow Constraints

    # All passengers are reassigned to some itinerary, which might be the same or different
    # from their originally scheduled itinerary (this include a null itinerary if reassignment
    # is not possible during the recovery period)
    pfc_1 = {
        (p, v): m.addConstr(quicksum(h[p, pd, v] for pd in CO_p[p]) == n[v][p])
        for p in range(len(P))
        for v in Y
    }

    # Passengers can be reassigned only to non-disrupted itineraries
    pfc_2 = {
        (p, v, pd): m.addConstr(h[p, pd, v] <= (1 - lambd[pd]) * n[v][p])
        for p in range(len(P))
        for v in Y
        for pd in CO_p[p]
    }

    # The number of passengers that do show up for their reassigned itineraries does not exceed
    # the total passenger capacity for their reassigned flight
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

    # Start time of arrival slot asl is no later than the combined scheduled arrival time and
    # arrival delay of flight f, only if the arrival slot is assigned to flight f.
    asc_1 = {
        (f, asl): m.addConstr(
            AA[asl][0] <= sta[f] + deltaA[f] + BIG_M * (1 - vA[asl, f])
        )
        for f in F
        for asl in AAF[f]
    }

    # End time of arrival slot asl is no earlier than the combined scheduled arrival time and
    # arrival delay of flight f, only if the arrival slot is assigned to flight f.
    asc_2 = {
        (f, asl): m.addConstr(
            AA[asl][1] >= sta[f] + deltaA[f] - BIG_M * (1 - vA[asl, f])
        )
        for f in F
        for asl in AAF[f]
    }

    # Each non-cancelled flight is assigned exactly one arrival slot
    asc_3 = {
        f: m.addConstr(quicksum(vA[asl, f] for asl in AAF[f]) == 1 - z[f]) for f in F
    }

    # Arrival slot capacity limit
    asc_4 = {
        asl: m.addConstr(quicksum(vA[asl, f] for f in FAA[asl]) <= scA[asl])
        for asl in range(len(AA))
    }

    # Start time of departure slot asl is no later than the combined scheduled departure time and
    # departure delay of flight f, only if the departure slot is assigned to flight f.
    asc_5 = {
        (f, dsl): m.addConstr(
            DA[dsl][0] <= std[f] + deltaD[f] + BIG_M * (1 - vD[dsl, f])
        )
        for f in F
        for dsl in DAF[f]
    }

    # End time of departure slot asl is no earlier than the combined scheduled departure time and
    # departure delay of flight f, only if the departure slot is assigned to flight f.
    asc_6 = {
        (f, dsl): m.addConstr(
            DA[dsl][1] >= std[f] + deltaD[f] - BIG_M * (1 - vD[dsl, f])
        )
        for f in F
        for dsl in DAF[f]
    }

    # Each non-cancelled flight is assigned exactly one departure slot
    asc_7 = {
        f: m.addConstr(quicksum(vD[dsl, f] for dsl in DAF[f]) == 1 - z[f]) for f in F
    }

    # Departure slot capacity limit
    asc_8 = {
        dsl: m.addConstr(quicksum(vD[dsl, f] for f in FDA[dsl]) <= scD[dsl])
        for dsl in range(len(DA))
    }

    # Flight Delay Constraints

    # relate the departure and arrival delays of each flight via delay absorption through
    # increased cruise speed.
    fdc_1 = {f: m.addConstr(deltaA[f] >= deltaD[f] - gamma[f] - sb[f]) for f in F}

    # relate the arrival delay of one flight to the departure delay of the next flight
    # operated by the same tail, by accounting for delay propagation.

    fdc_2 = {
        (f, fd, t): m.addConstr(
            deltaD[fd]
            >= deltaA[f]
            + mtt[f][fd][t]
            - ct[f][fd]
            - BIG_M * (3 - x[t, f] - x[t, fd] - y[f, fd])
        )
        for f in F
        for fd in CF_f[f]
        for t in list(set(T_f[f]).intersection(T_f[fd]))
    }

    # Itinerary Feasibility Constraints: Determine when an itinerary gets disrupted due
    # to flight cancelations and due to flight retiming decisions, respectively.
    ifc_1 = {
        (f, p): m.addConstr(lambd[p] >= z[f])
        for f in F
        for p in range(len(P))
        if f in P[p]
    }
    ifc_2 = {
        p: m.addConstr(
            std[CF_p[p][1]] + deltaD[CF_p[p][1]] - sta[CF_p[p][0]] - deltaA[CF_p[p][0]]
            >= mct[CF_p[p][0]][CF_p[p][1]][p] - BIG_M * lambd[p]
        )
        for p in range(len(P))
    }

    # Itinerary Delay Constraints

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

    print("\nList of assigned departure slots:")
    for f in F:
        for dsl in range(len(DA)):
            if vD[dsl, f].x > 0.9:
                print(f"Flight {f} assigned to departure slot {DA[dsl]}")

    print("\nList of assigned arrival slots:")
    for f in F:
        for asl in range(len(AA)):
            if vA[asl, f].x > 0.9:
                print(f"Flight {f} assigned to arrival slot {AA[asl]}")

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
        if rho[f].x > 0.9:
            print(f"Flight {f} is the first flight.")

    # print("\nSUS CONSTRAINTS:")
    # for f in F:
    #     for fd in CF_f[f]:
    #         for t in list(set(T_f[f]).intersection(T_f[fd])):
    #             import pdb; pdb.set_trace()
    #             print("1. 1+x[t,fd] >= x[t,f] + y[f,fd], 2. 1+x[t,f] >= x[t,fd] + y[f,fd]")
    #             print(f"1. 1+{x[t,fd].x}>={x[t,f].x}+{y[f,fd].x}, 2. {1}+{x[t,f].x}<= {x[t,fd].x}+{y[f,fd].x}")

    return


if __name__ == "__main__":
    airline_recovery_basic()
