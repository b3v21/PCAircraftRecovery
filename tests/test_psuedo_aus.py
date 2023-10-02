from gurobipy import *
from PC-aircraft-recovery.data.psuedo_aus_medium_size_data import *
from airline_recovery import *


def test_psuedo_aus_medium_size():
    m = Model("airline recovery basic")
    variables = generate_variables(m)
    x, _, _, _, _, _, h, _, _, deltaA, _, _, _, gamma, _, beta = variables

    set_objective(m, variables)
    flight_scheduling_constraints(m, variables)
    sequencing_and_fleet_size_constraints(m, variables)
    passenger_flow_constraints(m, variables)
    airport_slot_constraints(m, variables)
    flight_delay_constraints(m, variables)
    itinerary_feasibility_constraints(m, variables)
    itinerary_delay_constraints(m, variables)
    beta_linearizing_constraints(m, variables)

    print("optimizing to get xhat...")
    m.setParam("OutputFlag", 0)
    m.optimize()

    x_hat = generate_x_hat(m, variables)

    # Delay flight 0 by makings its arrival slot unavailable.
    AA.remove((54.0, 56.0))

    kappa = 1000

    # Generate new data
    AAF = {f: [i for i, slot in enumerate(AA) if sta[f] <= slot[0]] for f in F}
    DAF = {f: [i for i, slot in enumerate(DA) if std[f] <= slot[0]] for f in F}
    FAA = {asl: [f for f in F if sta[f] <= asl[1] and sta[f] >= asl[0]] for asl in AA}
    FDA = {dsl: [f for f in F if std[f] <= dsl[1] and std[f] >= dsl[0]] for dsl in DA}

    print("Regenerate neccecary constraints...")
    airport_slot_constraints(m, variables, (sta, std, AAF, DAF, FAA, FDA))
    itinerary_feasibility_constraints(m, variables, (sta, std, AAF, DAF, FAA, FDA))
    itinerary_delay_constraints(m, variables, (sta, std, AAF, DAF, FAA, FDA))
    set_objective(m, variables, (x_hat, kappa))

    print("optimizing...")
    m.setParam("OutputFlag", 1)
    m.optimize()

    print("generating output...")
    generate_output(m, variables)

    assert True
