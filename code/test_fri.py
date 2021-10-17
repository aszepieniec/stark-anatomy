from algebra import *
from fri import *

def test_fri( ):
    field = Field.main()
    degree = 63
    expansion_factor = 4
    num_colinearity_tests = 17

    initial_codeword_length = (degree + 1) * expansion_factor
    log_codeword_length = 0
    codeword_length = initial_codeword_length
    while codeword_length > 1:
        codeword_length //= 2
        log_codeword_length += 1

    assert(1 << log_codeword_length == initial_codeword_length), "log not computed correctly"

    omega = field.primitive_nth_root(initial_codeword_length)
    generator = field.generator()

    assert(omega^(1 << log_codeword_length) == field.one()), "omega not nth root of unity"
    assert(omega^(1 << (log_codeword_length-1)) != field.one()), "omega not primitive"

    fri = Fri(generator, omega, initial_codeword_length, expansion_factor, num_colinearity_tests)

    polynomial = Polynomial([FieldElement(i, field) for i in range(degree+1)])
    domain = [omega^i for i in range(initial_codeword_length)]

    codeword = polynomial.evaluate_domain(domain)

    # test valid codeword
    print("testing valid codeword ...")
    proof_stream = ProofStream()

    fri.prove(codeword, proof_stream)
    print("")
    points = []
    verdict = fri.verify(proof_stream, points)
    if verdict == False:
        print("rejecting proof, but proof should be valid!")
        return

    for (x,y) in points:
        if polynomial.evaluate(omega^x) != y:
            print("polynomial evaluates to wrong value")
            assert(False)
    print("success! \\o/")

    # disturb then test for failure
    print("testing invalid codeword ...")
    proof_stream = ProofStream()
    for i in range(0, degree//3):
        codeword[i] = field.zero()

    fri.prove(codeword, proof_stream)
    points = []
    assert False == fri.verify(proof_stream, points), "proof should fail, but is accepted ..."
    print("success! \\o/")

