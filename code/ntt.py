from univariate import *

def ntt( primitive_root, values ):
    assert(len(values) & (len(values) - 1) == 0), "cannot compute ntt of non-power-of-two sequence"
    if len(values) <= 1:
        return values

    field = values[0].field

    assert(primitive_root^len(values) == field.one()), "primitive root must be nth root of unity, where n is len(values)"
    assert(primitive_root^(len(values)//2) != field.one()), "primitive root is not primitive nth root of unity, where n is len(values)"

    half = len(values) // 2

    odds = ntt(primitive_root^2, values[1::2])
    evens = ntt(primitive_root^2, values[::2])

    return [evens[i % half] + (primitive_root^i) * odds[i % half] for i in range(len(values))]
 
def intt( primitive_root, values ):
    assert(len(values) & (len(values) - 1) == 0), "cannot compute intt of non-power-of-two sequence"

    if len(values) == 1:
        return values

    field = values[0].field
    ninv = FieldElement(len(values), field).inverse()

    transformed_values = ntt(primitive_root.inverse(), values)
    return [ninv*tv for tv in transformed_values]

def fast_multiply( lhs, rhs, primitive_root, root_order ):
    assert(primitive_root^root_order == primitive_root.field.one()), "supplied root does not have supplied order"
    assert(primitive_root^(root_order//2) != primitive_root.field.one()), "supplied root is not primitive root of supplied order"

    if lhs.is_zero() or rhs.is_zero():
        return Polynomial([])

    field = lhs.coefficients[0].field
    root = primitive_root
    order = root_order
    degree = lhs.degree() + rhs.degree()

    if degree < 8:
        return lhs * rhs

    while degree < order // 2:
        root = root^2
        order = order // 2

    lhs_coefficients = lhs.coefficients[:(lhs.degree()+1)]
    while len(lhs_coefficients) < order:
        lhs_coefficients += [field.zero()]
    rhs_coefficients = rhs.coefficients[:(rhs.degree()+1)]
    while len(rhs_coefficients) < order:
        rhs_coefficients += [field.zero()]

    lhs_codeword = ntt(root, lhs_coefficients)
    rhs_codeword = ntt(root, rhs_coefficients)

    hadamard_product = [l * r for (l, r) in zip(lhs_codeword, rhs_codeword)]

    product_coefficients = intt(root, hadamard_product)
    return Polynomial(product_coefficients[0:(degree+1)])

def fast_zerofier( domain, primitive_root, root_order ):
    assert(primitive_root^root_order == primitive_root.field.one()), "supplied root does not have supplied order"
    assert(primitive_root^(root_order//2) != primitive_root.field.one()), "supplied root is not primitive root of supplied order"

    if len(domain) == 0:
        return Polynomial([])

    if len(domain) == 1:
        return Polynomial([-domain[0], primitive_root.field.one()])

    half = len(domain) // 2

    left = fast_zerofier(domain[:half], primitive_root, root_order)
    right = fast_zerofier(domain[half:], primitive_root, root_order)
    return fast_multiply(left, right, primitive_root, root_order)

def fast_evaluate( polynomial, domain, primitive_root, root_order ):
    assert(primitive_root^root_order == primitive_root.field.one()), "supplied root does not have supplied order"
    assert(primitive_root^(root_order//2) != primitive_root.field.one()), "supplied root is not primitive root of supplied order"

    if len(domain) == 0:
        return []

    if len(domain) == 1:
        return [polynomial.evaluate(domain[0])]

    half = len(domain) // 2

    left_zerofier = fast_zerofier(domain[:half], primitive_root, root_order)
    right_zerofier = fast_zerofier(domain[half:], primitive_root, root_order)

    left = fast_evaluate(polynomial % left_zerofier, domain[:half], primitive_root, root_order)
    right = fast_evaluate(polynomial % right_zerofier, domain[half:], primitive_root, root_order)

    return left + right

def fast_interpolate( domain, values, primitive_root, root_order ):
    assert(primitive_root^root_order == primitive_root.field.one()), "supplied root does not have supplied order"
    assert(primitive_root^(root_order//2) != primitive_root.field.one()), "supplied root is not primitive root of supplied order"
    assert(len(domain) == len(values)), "cannot interpolate over domain of different length than values list"

    if len(domain) == 0:
        return Polynomial([])

    if len(domain) == 1:
        return Polynomial([values[0]])

    half = len(domain) // 2

    left_zerofier = fast_zerofier(domain[:half], primitive_root, root_order)
    right_zerofier = fast_zerofier(domain[half:], primitive_root, root_order)

    left_offset = fast_evaluate(right_zerofier, domain[:half], primitive_root, root_order)
    right_offset = fast_evaluate(left_zerofier, domain[half:], primitive_root, root_order)

    if not all(not v.is_zero() for v in left_offset):
        print("left_offset:", " ".join(str(v) for v in left_offset))

    left_targets = [n / d for (n,d) in zip(values[:half], left_offset)]
    right_targets = [n / d for (n,d) in zip(values[half:], right_offset)]

    left_interpolant = fast_interpolate(domain[:half], left_targets, primitive_root, root_order)
    right_interpolant = fast_interpolate(domain[half:], right_targets, primitive_root, root_order)

    return left_interpolant * right_zerofier + right_interpolant * left_zerofier

def fast_coset_evaluate( polynomial, offset, generator, order ):
    scaled_polynomial = polynomial.scale(offset)
    values = ntt(generator, scaled_polynomial.coefficients + [offset.field.zero()] * (order - len(polynomial.coefficients)))
    return values

def fast_coset_divide( lhs, rhs, offset, primitive_root, root_order ): # clean division only!
    assert(primitive_root^root_order == primitive_root.field.one()), "supplied root does not have supplied order"
    assert(primitive_root^(root_order//2) != primitive_root.field.one()), "supplied root is not primitive root of supplied order"
    assert(not rhs.is_zero()), "cannot divide by zero polynomial"

    if lhs.is_zero():
        return Polynomial([])

    assert(rhs.degree() <= lhs.degree()), "cannot divide by polynomial of larger degree"

    field = lhs.coefficients[0].field
    root = primitive_root
    order = root_order
    degree = max(lhs.degree(),rhs.degree())

    if degree < 8:
        return lhs / rhs

    while degree < order // 2:
        root = root^2
        order = order // 2

    scaled_lhs = lhs.scale(offset)
    scaled_rhs = rhs.scale(offset)
    
    lhs_coefficients = scaled_lhs.coefficients[:(lhs.degree()+1)]
    while len(lhs_coefficients) < order:
        lhs_coefficients += [field.zero()]
    rhs_coefficients = scaled_rhs.coefficients[:(rhs.degree()+1)]
    while len(rhs_coefficients) < order:
        rhs_coefficients += [field.zero()]

    lhs_codeword = ntt(root, lhs_coefficients)
    rhs_codeword = ntt(root, rhs_coefficients)

    quotient_codeword = [l / r for (l, r) in zip(lhs_codeword, rhs_codeword)]
    scaled_quotient_coefficients = intt(root, quotient_codeword)
    scaled_quotient = Polynomial(scaled_quotient_coefficients[:(lhs.degree() - rhs.degree() + 1)])

    return scaled_quotient.scale(offset.inverse())

