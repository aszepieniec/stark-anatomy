from algebra import *
from univariate import *
from multivariate import *
from rescue_prime import *
from fri import *
from ip import *
from fast_stark import *

def test_fast_stark( ):
    field = Field.main()
    expansion_factor = 4
    num_colinearity_checks = 2
    security_level = 2

    rp = RescuePrime()
    output_element = field.sample(bytes(b'0xdeadbeef'))

    for trial in range(0, 20):
        input_element = output_element
        print("running trial with input:", input_element.value)
        output_element = rp.hash(input_element)
        num_cycles = rp.N+1
        state_width = rp.m

        stark = FastStark(field, expansion_factor, num_colinearity_checks, security_level, state_width, num_cycles)
        transition_zerofier, transition_zerofier_codeword, transition_zerofier_root = stark.preprocess()

        # prove honestly
        print("honest proof generation ...")

        # prove
        trace = rp.trace(input_element)
        air = rp.transition_constraints(stark.omicron)
        boundary = rp.boundary_constraints(output_element)
        proof = stark.prove(trace, air, boundary, transition_zerofier, transition_zerofier_codeword)

        # verify
        verdict = stark.verify(proof, air, boundary, transition_zerofier_root)

        assert(verdict == True), "valid stark proof fails to verify"
        print("success \\o/")

        print("verifying false claim ...")
        # verify false claim
        output_element_ = output_element + field.one()
        boundary_ = rp.boundary_constraints(output_element_)
        verdict = stark.verify(proof, air, boundary_, transition_zerofier_root)

        assert(verdict == False), "invalid stark proof verifies"
        print("proof rejected! \\o/")

        # prove with false witness
        print("attempting to prove with witness violating transition constraints (should not fail because using fast division) ...")
        cycle = 1 + (int(os.urandom(1)[0]) % len(trace)-1)
        register = int(os.urandom(1)[0]) % state_width
        error = field.sample(os.urandom(17))
    
        trace[cycle][register] = trace[cycle][register] + error
    
        proof = stark.prove(trace, air, boundary, transition_zerofier, transition_zerofier_codeword)

        print(" ... but verification should fail :D")
        verdict = stark.verify(proof, air, boundary, transition_zerofier_root)
        assert(verdict == False), "STARK produced from false witness verifies :("
        print("proof rejected! \\o/")

