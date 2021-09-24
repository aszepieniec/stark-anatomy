from rescue_prime import *
import os

def test_rescue_prime( ):
    rp = RescuePrime()
    
    # test vectors
    assert(rp.hash(FieldElement(1, rp.field)) == FieldElement(244180265933090377212304188905974087294, rp.field)), "rescue prime test vector 1 failed"
    assert(rp.hash(FieldElement(57322816861100832358702415967512842988, rp.field)) == FieldElement(89633745865384635541695204788332415101, rp.field)), "rescue prime test vector 2 failed"


    # test trace boundaries
    a = FieldElement(57322816861100832358702415967512842988, rp.field)
    b = FieldElement(89633745865384635541695204788332415101, rp.field)
    trace = rp.trace(a)
    assert(trace[0][0] == a and trace[-1][0] == b), "rescue prime trace does not satisfy boundary conditions"

    print("Rescue-Prime eval tests pass \\o/")

def test_trace( ):
    rp = RescuePrime()

    input_element = FieldElement(57322816861100832358702415967512842988, rp.field)
    b = FieldElement(89633745865384635541695204788332415101, rp.field)
    output_element = rp.hash(input_element)
    assert(b == output_element), "output elements do not match"

    # get trace
    trace = rp.trace(input_element)

    # test boundary constraints
    for condition in rp.boundary_constraints(output_element):
        cycle, element, value = condition
        if trace[cycle][element] != value:
            print("rescue prime boundary condition error: trace element", element, "at cycle", cycle, "has value", trace[cycle][element], "but should have value", value)
            assert(False)

    # test transition constraints
    omicron = rp.field.primitive_kth_root(1 << 119)
    transition_constraints = rp.transition_constraints(omicron)
    first_step_constants, second_step_constants = rp.round_constants_polynomials(omicron)
    for o in range(len(trace)-1):
        for air_poly in rp.transition_constraints(omicron):
            previous_state = [trace[o][0], trace[o][1]]
            next_state = [trace[o+1][0], trace[o+1][1]]
            point = [omicron^o] + previous_state + next_state
            if air_poly.evaluate(point) != rp.field.zero():
                assert(False), "air polynomial does not evaluate to zero"

    print("valid Rescue-Prime trace passes tests, testing invalid traces ...")

    # insert errors into trace, to make sure errors get noticed
    for k in range(10):
        print("trial", k, "...")
        # sample error location and value randomly
        register_index = int(os.urandom(1)[0]) % rp.m
        cycle_index = int(os.urandom(1)[0]) % (rp.N+1)
        value_ = rp.field.sample(os.urandom(17))
        if value_ == rp.field.zero():
            continue

        # reproduce deterministic error
        if k == 0:
            register_index = 1
            cycle_index = 22
            value_ = FieldElement(17274817952119230544216945715808633996, rp.field)

        # perturb
        trace[cycle_index][register_index] = trace[cycle_index][register_index] + value_
    
        error_got_noticed = False

        # test boundary constraints
        for condition in rp.boundary_constraints(output_element):
            if error_got_noticed:
                break
            cycle, element, value = condition
            if trace[cycle][element] != value:
                error_got_noticed = True

        # test transition constraints
        for o in range(len(trace)-1):
            if error_got_noticed:
                break
            for air_poly in rp.transition_constraints(omicron):
                previous_state = [trace[o][0], trace[o][1]]
                next_state = [trace[o+1][0], trace[o+1][1]]
                point = [omicron^o] + previous_state + next_state
                if air_poly.evaluate(point) != rp.field.zero():
                    error_got_noticed = True

        # if error was not noticed, panic
        if not error_got_noticed:
            print("error was not noticed.")
            print("register index:", register_index)
            print("cycle index:", cycle_index)
            print("value_:", value_)
            assert(False), "error was not noticed"

        trace[cycle_index][register_index] = trace[cycle_index][register_index] - value_

    print("Rescue-Prime trace tests pass \\o/")

