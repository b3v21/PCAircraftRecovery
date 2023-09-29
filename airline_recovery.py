from gurobipy import *
from data.test_pseudo_aus import *

BIG_M = 999999999

# QUESTIONS: is rho redundant?


def run_aircraft_recovery() -> None:
    """
    basic initial implementation of Passenger-Centric Integrated Airline Schedule and
    Aircraft Recovery.

    Uses data imported from the file imported above.
    """

    m = Model("airline recovery basic")
    variables = generate_variables(m)
    x, _, _, _, _, _, h, _, _, deltaA, _, _, _, gamma, _, beta = variables

    m.setObjective(
        (
            quicksum(oc[t, f] * x[t, f] for t in T for f in F_t[t])
            + quicksum(fc[f] * gamma[f] for f in F)
            + quicksum(dc[f] * deltaA[f] for f in F)
            + quicksum(
                rc[P.index(p), pd]
                * (
                    h[P.index(p), pd, v]
                    - quicksum(
                        theta[v, P.index(p), pd, g] * beta[v, P.index(p), pd, g]
                        for g in Z
                    )
                )
                for v in Y
                for p in P
                for pd in CO_p[P.index(p)]
            )
            + quicksum(
                pc[g, P.index(p), pd] * beta[v, P.index(p), pd, g]
                for v in Y
                for p in P
                for pd in CO_p[P.index(p)]
                for g in Z
            )
            + kappa
            * quicksum(
                x[t, f] - 2 * x_hat[f, t] * x[t, f] + x_hat[f, t]
                for t in T
                for f in F_t[t]
            )
        ),
        GRB.MINIMIZE,
    )

    flight_scheduling_constraints(m, variables)
    sequencing_and_fleet_size_constraints(m, variables)
    passenger_flow_constraints(m, variables)
    airport_slot_constraints(m, variables)
    flight_delay_constraints(m, variables)
    itinerary_feasibility_constraints(m, variables)
    # itinerary_delay_constraints(m, variables)
    # beta_linearizing_constraints(m, variables)

    m.optimize()

    generate_output(m, variables)


def generate_variables(m: Model) -> list[dict[list[int], Var]]:
    """
    Generate variables for the model
    """
    # Variables
    # 1 if tail t is assigned to flight f
    x = {(t, f): m.addVar(vtype=GRB.BINARY) for t in T for f in F}

    # 1 if flight f is cancelled
    z = {f: m.addVar(vtype=GRB.BINARY) for f in F}

    #  if flight f is flown and then flight fd is flown
    y = {(f, fd): m.addVar(vtype=GRB.BINARY) for f in F for fd in F if fd != f}

    # 1 if flight f is the last flight in the recovery period operated by its tail
    sigma = {f: m.addVar(vtype=GRB.BINARY) for f in F}

    #  if flight f is the first flight in the recovery period operated by its tail
    rho = {f: m.addVar(vtype=GRB.BINARY) for f in F}

    # 1 if flight f is the first flight for tail t in the recovery period
    phi = {(t, f): m.addVar(vtype=GRB.BINARY) for f in F for t in T}

    # number of passengers in fare class v, reassigned from itinerary p to itinerary p
    h = {
        (p, pd, v): m.addVar() for v in Y for p in range(len(P)) for pd in range(len(P))
    }

    # lambd[p] = 1 if itinerary p is disrupted
    lambd = {p: m.addVar(vtype=GRB.BINARY) for p in range(len(P))}

    # 1 if itinerary p is reassigned to itinerary pd with delay level g
    alpha = {
        (p, pd, g): m.addVar(vtype=GRB.BINARY)
        for p in range(len(P))
        for pd in range(len(P))
        for g in Z
    }

    # arrival delay of flight f
    deltaA = {f: m.addVar() for f in F}

    # departure delay of flight f
    deltaD = {f: m.addVar() for f in F}

    # 1 if arrival slot asl is assigned to flight f
    vA = {(asl, f): m.addVar(vtype=GRB.BINARY) for f in F for asl in range(len(AA))}

    # 1 if departure slot dsl is assigned to flight f
    vD = {(dsl, f): m.addVar(vtype=GRB.BINARY) for f in F for dsl in range(len(DA))}

    # delay absorbed by flight f
    gamma = {f: m.addVar() for f in F}

    # Arrival delay of itinerary pd with respect to planned arrival time of itinerary p
    tao = {(p, pd): m.addVar(lb=-GRB.INFINITY) for p in range(len(P)) for pd in CO_p[p]}

    # number of passengers in fare class v with originally scheduled itinerary p,
    # reassigned to itinerary pd, corresponding to an arrival delay level ζ
    beta = {
        (v, p, pd, g): m.addVar(vtype=GRB.INTEGER)
        for v in Y
        for p in range(len(P))
        for pd in CO_p[p]
        for g in Z
    }

    return [
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
    ]


def flight_scheduling_constraints(
    m: Model, variables: list[dict[list[int], Var]]
) -> None:
    """
    Flight Scheduling Constraints
    """

    x, z, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = variables

    # Every flight must be either flown using exactly one aircraft or must be cancelled
    fsc_1 = {f: m.addConstr(quicksum(x[t, f] for t in T_f[f]) + z[f] == 1) for f in F}


def sequencing_and_fleet_size_constraints(
    m: Model, variables: list[dict[list[int], Var]]
) -> None:
    """
    Sequencing & Fleet Size Constraints

    NOTE: Sequencing & Fleet Size Constraints still allow for a tail to not be used
    at all during the recovery period
    """

    x, z, y, sigma, rho, phi, _, _, _, _, _, _, _, _, _, _ = variables

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
            <= tb[t, k]
        )
        for t in T
        for k in K
    }


def passenger_flow_constraints(m: Model, variables: list[dict[list[int], Var]]) -> None:
    """
    Passenger Flow Constraints
    """

    x, _, _, _, _, _, h, lambd, _, _, _, _, _, _, _, beta = variables

    # All passengers are reassigned to some itinerary, which might be the same or
    # different from their originally scheduled itinerary (this include a null itinerary
    # if reassignment is not possible during the recovery period)
    pfc_1 = {
        (P.index(p), v): m.addConstr(
            quicksum(h[P.index(p), pd, v] for pd in CO_p[P.index(p)])
            == n[v, P.index(p)]
        )
        for p in P
        for v in Y
    }

    # Passengers can be reassigned only to non-disrupted itineraries
    pfc_2 = {
        (P.index(p), v, pd): m.addConstr(
            h[P.index(p), pd, v] <= (1 - lambd[pd]) * n[v, P.index(p)]
        )
        for p in P
        for v in Y
        for pd in CO_p[P.index(p)]
    }

    # The number of passengers that do show up for their reassigned itineraries does not
    # exceed the total passenger capacity for their reassigned flight
    pfc_3 = {
        f: m.addConstr(
            quicksum(q[t] * x[t, f] for t in T_f[f])
            >= quicksum(
                (
                    h[P.index(p), pd, v]
                    - quicksum(
                        theta[v, P.index(p), pd, g] * beta[v, P.index(p), pd, g]
                        for g in Z
                    )
                )
                for v in Y
                for p in P
                for pd in CO_p[P.index(p)]
                if f in P[pd]
            )
        )
        for f in F
    }


def airport_slot_constraints(m: Model, variables: list[dict[list[int], Var]]) -> None:
    """
    Airport slot constraints
    """

    _, z, _, _, _, _, _, _, _, deltaA, deltaD, vA, vD, _, _, _ = variables

    # Start time of arrival slot asl is no later than the combined scheduled arrival time
    # and arrival delay of flight f, only if the arrival slot is assigned to flight f.
    asc_1 = {
        (f, asl): m.addConstr(
            AA[asl][0] <= sta[f] + deltaA[f] + BIG_M * (1 - vA[asl, f])
        )
        for f in F
        for asl in AAF[f]
    }

    # End time of arrival slot asl is no earlier than the combined scheduled arrival time
    # and arrival delay of flight f, only if the arrival slot is assigned to flight f.
    asc_2 = {
        (f, asl): m.addConstr(
            AA[asl][1] >= sta[f] + deltaA[f] - BIG_M * (1 - vA[asl, f])
        )
        for f in F
        for asl in AAF[f]
    }

    # # Each non-cancelled flight is assigned exactly one arrival slot
    asc_3 = {
        f: m.addConstr(quicksum(vA[asl, f] for asl in AAF[f]) == 1 - z[f]) for f in F
    }

    # Arrival slot capacity limit
    asc_4 = {
        asl: m.addConstr(quicksum(vA[AA.index(asl), f] for f in FAA[asl]) <= scA[asl])
        for asl in AA
    }

    # Start time of departure slot asl is no later than the combined scheduled departure
    # time and departure delay of flight f, only if the departure slot is assigned to
    # flight f.
    asc_5 = {
        (f, dsl): m.addConstr(
            DA[dsl][0] <= std[f] + deltaD[f] + BIG_M * (1 - vD[dsl, f])
        )
        for f in F
        for dsl in DAF[f]
    }

    # End time of departure slot asl is no earlier than the combined scheduled departure
    # time and departure delay of flight f, only if the departure slot is assigned to
    # flight f.
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

    # # Departure slot capacity limit
    asc_8 = {
        dsl: m.addConstr(quicksum(vD[DA.index(dsl), f] for f in FDA[dsl]) <= scD[dsl])
        for dsl in DA
    }


def flight_delay_constraints(m: Model, variables: list[dict[list[int], Var]]) -> None:
    """
    Flight Delay Constraints
    """

    x, _, y, _, _, _, _, _, _, deltaA, deltaD, _, _, gamma, _, _ = variables

    # relate the departure and arrival delays of each flight via delay absorption through
    # increased cruise speed.
    fdc_1 = {f: m.addConstr(deltaA[f] >= deltaD[f] - gamma[f] - sb[f]) for f in F}

    # relate the arrival delay of one flight to the departure delay of the next flight
    # operated by the same tail, by accounting for delay propagation.

    fdc_2 = {
        (f, fd, t): m.addConstr(
            deltaD[fd]
            >= deltaA[f]
            + mtt[t, f, fd]
            - ct[f, fd]
            - BIG_M * (3 - x[t, f] - x[t, fd] - y[f, fd])
        )
        for f in F
        for fd in CF_f[f]
        for t in list(set(T_f[f]).intersection(set(T_f[fd])))
    }


def itinerary_feasibility_constraints(
    m: Model, variables: list[dict[list[int], Var]]
) -> None:
    """
    Itinerary Feasibility Constraints: Determine when an itinerary gets disrupted due
    to flight cancelations and due to flight retiming decisions, respectively.
    """

    _, z, _, _, _, _, _, lambd, _, deltaA, deltaD, _, _, _, _, _ = variables

    ifc_1 = {
        (f, p): m.addConstr(lambd[p] >= z[f])
        for f in F
        for p in range(len(P))
        if f in P[p]
    }

    ifc_2 = {
        (P.index(p), (f, fd)): m.addConstr(
            std[fd] + deltaD[fd] - sta[f] - deltaA[f]
            >= mct[P.index(p), f, fd] - BIG_M * lambd[P.index(p)]
        )
        for p in P
        for (f, fd) in CF_p[P.index(p)]
        if CF_p[P.index(p)] != []
    }


def itinerary_delay_constraints(
    m: Model, variables: list[dict[list[int], Var]]
) -> None:
    """
    Itinerary Delay Constraints
    """

    _, _, _, _, _, _, _, _, alpha, deltaA, _, _, _, _, tao, _ = variables

    # Calculate the passenger arrival delay after the itinerary reassignment.
    idc_1 = {
        (p, pd): m.addConstr(
            tao[p, pd]
            == quicksum(lf[pd,fd] * (deltaA[fd] + sta[fd]) for fd in F)
            - quicksum(lf[p,f] * sta[f] for f in F)
        )
        for p in range(len(P))
        for pd in CO_p[p]
    }

    # Determine the passenger delay level for each pair of compatible itineraries,
    # depending on the actual delay value in minutes.
    idc_2 = {
        (p, pd): m.addConstr(
            tao[p, pd] <= quicksum(small_theta[g] * alpha[p, pd, g] for g in Z)
        )
        for p in range(len(P))
        for pd in CO_p[p]
    }
    idc_3 = {
        (p, pd): m.addConstr(quicksum(alpha[p, pd, g] for g in Z) == 1)
        for p in range(len(P))
        for pd in CO_p[p]
    }


def beta_linearizing_constraints(
    m: Model, variables: list[dict[list[int], Var]]
) -> None:
    """
    Constraints which allow beta to behave like alpha * h, while being linear
    """

    _, _, _, _, _, _, h, _, alpha, _, _, _, _, _, _, beta = variables

    blc_1 = {
        (v, p, pd, g): m.addConstr(
            beta[v, p, pd, g] <= h[p, pd, v] + BIG_M * (1 - alpha[p, pd, g])
        )
        for v in Y
        for p in range(len(P))
        for pd in CO_p[p]
        for g in Z
    }
    blc_2 = {
        (v, p, pd, g): m.addConstr(
            beta[v, p, pd, g] >= h[p, pd, v] - BIG_M * (1 - alpha[p, pd, g])
        )
        for v in Y
        for p in range(len(P))
        for pd in CO_p[p]
        for g in Z
    }
    blc_3 = {
        (v, p, pd, g): m.addConstr(beta[v, p, pd, g] <= BIG_M * alpha[p, pd, g])
        for v in Y
        for p in range(len(P))
        for pd in CO_p[p]
        for g in Z
    }


def generate_output(m: Model, variables: list[dict[list[int], Var]]) -> None:
    x, z, y, sigma, rho, phi, h, lambd, _, _, _, vA, vD, _, _, _ = variables
    print(72 * "-")
    print("\nTail to Flight Assignments:")
    for f in F:
        found_chained_flight = False
        for fd in CF_f[f]:
            if fd != f:
                if y[f, fd].x > 0.9:
                    for t in T:
                        if x[t, f].x > 0.9 and x[t, fd].x > 0.9:
                            found_chained_flight = True
                            print(f"T{t}: F{f} -> F{fd}")
                            if found_chained_flight:
                                break
                    if found_chained_flight:
                        break
        if not found_chained_flight:
            for t in T:
                if x[t, f].x > 0.9 and (sigma[f].x < 0.9 or rho[f].x > 0.9):
                    for dsl in range(len(DA)):
                        for asl in range(len(AA)):
                            if vD[dsl, f].x > 0.9 and vA[asl, f].x > 0.9:
                                if t < 10 and f < 10:
                                    print(f"T0{t}: F0{f} \t{DK_f[f]} --> {AK_f[f]} \tFlight Slots: {DA[dsl]} --> {AA[asl]}")
                                elif t < 10:
                                    print(f"T0{t}: F{f} \t{DK_f[f]} --> {AK_f[f]} \tFlight Slots: {DA[dsl]} --> {AA[asl]}")
                                elif f < 10:
                                    print(f"T{t}: F0{f} \t{DK_f[f]} --> {AK_f[f]} \tFlight Slots: {DA[dsl]} --> {AA[asl]}")
                                else:
                                    print(f"T{t}: F{f} \t{DK_f[f]} --> {AK_f[f]} \tFlight Slots: {DA[dsl]} --> {AA[asl]}")

    cancelled = False
    print("\nCancelled Flights:")
    for f in F:
        if z[f].x > 0.9:
            print(f"F{f} cancelled")
            cancelled = True
    if not cancelled:
        print("No Flights Cancelled")

    print("\nDisrupted Itineraries:")
    for p in range(len(P)):
        if lambd[p].x > 0.9:
            print(f"I{p} disrupted:")

            for pd in range(len(P)):
                for v in Y:
                    if p != pd:
                        if int(h[p, pd, v].x) > 0:
                            print(
                                f"    I{p} {*P[p],} -> I{pd} {*P[pd],} (fare class: {v}) people reassigned: {int(h[p, pd, v].x)}"
                            )

    print("\n" + 72 * "-")


if __name__ == "__main__":
    run_aircraft_recovery()
