from multivariate import *

def test_evaluate( ):
    field = Field.main()
    variables = MPolynomial.variables(4, field)
    zero = field.zero()
    one = field.one()
    two = FieldElement(2, field)
    five = FieldElement(5, field)

    mpoly1 = MPolynomial.constant(one) * variables[0] + MPolynomial.constant(two) * variables[1] + MPolynomial.constant(five) * (variables[2]^3)
    mpoly2 = MPolynomial.constant(one) * variables[0] * variables[3] + MPolynomial.constant(five) * (variables[3]^3) + MPolynomial.constant(five)

    mpoly3 = mpoly1 * mpoly2

    point = [zero, five, five, two]

    eval1 = mpoly1.evaluate(point)
    eval2 = mpoly2.evaluate(point)
    eval3 = mpoly3.evaluate(point)

    assert(eval1 * eval2 == eval3), "multivariate polynomial multiplication does not commute with evaluation"
    assert(eval1 + eval2 == (mpoly1 + mpoly2).evaluate(point)), "multivariate polynomial addition does not commute with evaluation"

    print("eval3:", eval3.value)
    print("multivariate evaluate test success \\o/")

def test_lift( ):
    field = Field.main()
    variables = MPolynomial.variables(4, field)
    zero = field.zero()
    one = field.one()
    two = FieldElement(2, field)
    five = FieldElement(5, field)

    upoly = Polynomial.interpolate_domain([zero, one, two], [two, five, five])
    mpoly = MPolynomial.from_univariate(upoly, 3)

    assert(upoly.evaluate(five) == mpoly.evaluate([zero, zero, zero, five])), "lifting univariate to multivariate failed"

    print("lifting univariate to multivariate polynomial success \\o/")
