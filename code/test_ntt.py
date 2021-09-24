from algebra import *
from univariate import *
from ntt import *
import os

def test_ntt( ):
    field = Field.main()
    logn = 8
    n = 1 << logn
    primitive_root = field.primitive_nth_root(n)

    coefficients = [field.sample(os.urandom(17)) for i in range(n)]
    poly = Polynomial(coefficients)

    values = ntt(primitive_root, coefficients)

    values_again = poly.evaluate_domain([primitive_root^i for i in range(len(values))])

    assert(values == values_again), "ntt does not compute correct batch-evaluation"

def test_intt( ):
    field = Field.main()

    logn = 7
    n = 1 << logn
    primitive_root = field.primitive_nth_root(n)

    values = [field.sample(os.urandom(1)) for i in range(n)]
    coeffs = ntt(primitive_root, values)
    values_again = intt(primitive_root, coeffs)

    assert(values == values_again), "inverse ntt is different from forward ntt"

def test_multiply( ):
    field = Field.main()

    logn = 6
    n = 1 << logn
    primitive_root = field.primitive_nth_root(n)

    for trial in range(20):
        lhs_degree = int(os.urandom(1)[0]) % (n // 2)
        rhs_degree = int(os.urandom(1)[0]) % (n // 2)

        lhs = Polynomial([field.sample(os.urandom(17)) for i in range(lhs_degree+1)])
        rhs = Polynomial([field.sample(os.urandom(17)) for i in range(rhs_degree+1)])

        fast_product = fast_multiply(lhs, rhs, primitive_root, n)
        slow_product = lhs * rhs

        assert(fast_product == slow_product), "fast product does not equal slow product"

def test_zerofier( ):
    field = Field.main()

    logn = 12
    n = 1 << logn
    primitive_root = field.primitive_nth_root(n)

    for trial in range(20):
        rand1 = int(os.urandom(1)[0]) % n
        rand2 = int(os.urandom(1)[0]) % n
        zeros_start = min(rand1, rand2)
        zeros_stop = max(rand1, rand2)
        if zeros_start == zeros_stop:
            continue
        zerofier_codeword = fast_zerofier_codeword(primitive_root, n, zeros_start, zeros_stop)
        assert(zerofier_codeword[zeros_start:zeros_stop] == [field.zero()] * (zeros_stop - zeros_start)), "zero points are not zero in zerofier codeword"
        assert(z != field.zero() for z in zerofier_codeword[:zeros_start]), "zerofier codeword has zeros before zeros start"
        assert(z != field.zero() for z in zerofier_codeword[zeros_stop:]), "zerofier codeword has zeros after zeros stop"

def test_interpolate( ):
    field = Field.main()

    logn = 9
    n = 1 << logn
    primitive_root = field.primitive_nth_root(n)

    for trial in range(10):
        N = sum((1 << (8*i)) * int(os.urandom(1)[0]) for i in range(8)) % n
        if N == 0:
            continue
        print("N:", N)
        values = [field.sample(os.urandom(1)) for i in range(N)]
        poly = fast_interpolate(primitive_root, n, values)
        print("poly degree:", poly.degree())
        values_again = fast_evaluate(poly, primitive_root, n)[0:N]
        values_again = poly.evaluate_domain([primitive_root^i for i in range(N)])

        if values != values_again:
            print("fast interpolation and evaluation are not inverses")
            print("expected:", ",".join(str(c.value) for c in values))
            print("observed:", ",".join(str(c.value) for c in values_again))
            assert(False)
        print("")

