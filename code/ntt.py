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

def fast_zerofier_codeword( primitive_root, root_order, zeros_start, zeros_stop ):
    field = primitive_root.field
    assert(primitive_root^root_order == field.one()), "root order argument does not match actual root's order"
    assert(primitive_root^(root_order // 2) != field.one()), "primitive root has lower order than indicated"
    num_zeros = zeros_stop - zeros_start

    # if zerofier has no zeros, it is the constant one polynomial
    if num_zeros == 0:
        return [field.one()] * root_order

    # if we only have a single zero, evaluate directly
    if num_zeros == 1:
        zero = primitive_root^zeros_start
        omegai = field.one()
        values = [field.zero()] * root_order
        for i in range(root_order):
            values[i] = omegai - zero
            omegai = omegai * primitive_root
        return values

    # if we don't have an even number of zeros, treat the last one separately
    if num_zeros % 2 == 1:
        values = fast_zerofier_codeword(primitive_root, root_order, zeros_start, zeros_stop-1)
        last_zero = primitive_root^(zeros_stop-1)
        omegai = field.one()
        for i in range(root_order):
            values[i] = values[i] * (omegai - last_zero)
            omegai = omegai * primitive_root
        return values

    # if the number of zeros is divisible by two, recurse on half
    half = num_zeros // 2
    half_zerofier_codeword = fast_zerofier_codeword(primitive_root, root_order, zeros_start, zeros_start+half)
    values = [field.zero()] * root_order
    for i in range(root_order):
        values[i] = half_zerofier_codeword[i] * half_zerofier_codeword[(root_order + i - half) % root_order]
    return values

def fast_evaluate( polynomial, primitive_root, root_order ):
    assert(root_order & (root_order - 1) == 0), "root order must be a power of two for fast evaluation"
    assert(polynomial.degree() < root_order), f"polynomial cannot have degree larger than root order ({root_order})"

    if polynomial.is_zero():
        return [primitive_root.field.zero()] * root_order


    polynomial_coefficients = polynomial.coefficients[:(polynomial.degree()+1)]
    coefficients = polynomial_coefficients + [primitive_root.field.zero()] * (root_order - len(polynomial_coefficients))
    return ntt(primitive_root, coefficients)

def fast_interpolate( primitive_root, root_order, values ):
    assert(len(values) <= root_order), "cannot interpolate through more values than contained in span of primitive root"
    assert(root_order & (root_order - 1) == 0), "root order must be power of two"
    assert(len(values) > 0), "number of values to interpolate through must be greater than zero"

    field = values[0].field

    acc = Polynomial([])
    bit_indices = [i for i in range(len(bin(len(values))[2:])) if (len(values) & (1 << i) != 0)] # indices of set bits

    # if length is small, interpolate naively
    if len(values) < 4:
        return Polynomial.interpolate_domain([primitive_root^i for i in range(len(values))], values)

    # if length is power of two, decompose into two equal halves
    if len(bit_indices) == 1:
        half = len(values) // 2

        left_zerofier_codeword = fast_zerofier_codeword(primitive_root, root_order, 0, half)
        left_zerofier = Polynomial(intt(primitive_root, left_zerofier_codeword))
        right_target = [values[half+i] / left_zerofier_codeword[half+i] for i in range(half)]
        right_interpolant = fast_interpolate(primitive_root, root_order, right_target)
        right_interpolant = right_interpolant.scale(primitive_root.inverse()^half)
        right = fast_multiply(left_zerofier, right_interpolant, primitive_root, root_order)

        right_zerofier_codeword = fast_zerofier_codeword(primitive_root, root_order, half, len(values))
        right_zerofier = Polynomial(intt(primitive_root, right_zerofier_codeword))
        left_target = [values[i] / right_zerofier_codeword[i] for i in range(half)]
        left_interpolant = fast_interpolate(primitive_root, root_order, left_target)
        left = fast_multiply(right_zerofier, left_interpolant, primitive_root, root_order)

        return right + left

    # else, decompose into chunks of length 2^k
    else:
        print("indices of set bits:", bit_indices)
        chunk_sizes = [1 << bi for bi in bit_indices]
        print("sizes of chunks:", chunk_sizes)
        print("number of values:", len(values), " in binary:", bin(len(values)))
        print("sum of chunk sizes:", sum(chunk_sizes))

        # initialize accumulator polynomials
        interpolant_accumulator = Polynomial([])
        zerofier_accumulator = Polynomial([field.one()])

        # for each segment,
        for i in range(len(bit_indices)):
            index = bit_indices[i]
            current_chunk_size = chunk_sizes[i]
            current_chunk_start = sum(chunk_sizes[:i])
            current_chunk_stop = sum(chunk_sizes[:(i+1)])

            # find right slice of values
            values_slice = values[current_chunk_start:current_chunk_stop]

            # find errors, which are to be corrected
            current_values = fast_evaluate(interpolant_accumulator, primitive_root, root_order)
            errors = current_values[current_chunk_start:current_chunk_stop]

            # find value of zerofier
            zerofier_values = fast_evaluate(zerofier_accumulator, primitive_root, root_order)
            assert(zerofier_values[:current_chunk_start] == [field.zero()] * current_chunk_start), "predecessor chunks of zerofier accumulator should be zero"
            zerofiers = zerofier_values[current_chunk_start:current_chunk_stop]

            # find values to interpolate between
            # targets * zerofiers + errors = values_slice
            targets = [( values_slice[i] - errors[i] ) / zerofiers[i] for i in range(current_chunk_size)]

            # find target interpolant
            #interpolant = Polynomial.interpolate_domain([primitive_root^i for i in range(current_chunk_start, current_chunk_stop)], targets)
            interpolant = fast_interpolate(primitive_root, root_order, targets)
            interpolant = interpolant.scale(primitive_root.inverse()^current_chunk_start)

            # combine interpolant with zerofier, so that we can safely add without affecting previous values
            #safe_chunk_interpolant = interpolant * zerofier_accumulator
            safe_chunk_interpolant = fast_multiply(interpolant, zerofier_accumulator, primitive_root, root_order)

            # expand zerofier accumulator with zerofier for current chunk
            zerofier = Polynomial.zerofier_domain([primitive_root^i for i in range(current_chunk_start, current_chunk_stop)])
            #zerofier_accumulator = zerofier_accumulator * zerofier
            zerofier_accumulator = fast_multiply(zerofier_accumulator, zerofier, primitive_root, root_order)

            # accumulate interpolant for current chunkp
            interpolant_accumulator = interpolant_accumulator + safe_chunk_interpolant
            interpolant_codeword = [interpolant_accumulator.evaluate(primitive_root^i) for i in range(len(values))]

        return interpolant_accumulator

