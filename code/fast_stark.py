from fri import *
from univariate import *
from multivariate import *
from ntt import *
from functools import reduce
import os

class FastStark:
    def __init__( self, field, expansion_factor, num_colinearity_checks, security_level, num_registers, num_cycles, transition_constraints_degree=2 ):
        assert(len(bin(field.p)) - 2 >= security_level), "p must have at least as many bits as security level"
        assert(expansion_factor & (expansion_factor - 1) == 0), "expansion factor must be a power of 2"
        assert(expansion_factor >= 4), "expansion factor must be 4 or greater"
        assert(num_colinearity_checks * 2 >= security_level), "number of colinearity checks must be at least half of security level"

        self.field = field
        self.expansion_factor = expansion_factor
        self.num_colinearity_checks = num_colinearity_checks
        self.security_level = security_level

        self.num_randomizers = 4*num_colinearity_checks

        self.num_registers = num_registers
        self.original_trace_length = num_cycles
        
        self.randomized_trace_length = self.original_trace_length + self.num_randomizers
        self.omicron_domain_length = 1 << len(bin(self.randomized_trace_length * transition_constraints_degree)[2:])
        self.fri_domain_length = self.omicron_domain_length * expansion_factor

        self.generator = self.field.generator()
        self.omega = self.field.primitive_nth_root(self.fri_domain_length)
        self.omicron = self.field.primitive_nth_root(self.omicron_domain_length)
        self.omicron_domain = [self.omicron^i for i in range(self.omicron_domain_length)]

        self.fri = Fri(self.generator, self.omega, self.fri_domain_length, self.expansion_factor, self.num_colinearity_checks)

    def preprocess( self ):
        transition_zerofier = fast_zerofier(self.omicron_domain[:(self.original_trace_length-1)], self.omicron, len(self.omicron_domain))
        transition_zerofier_codeword = fast_coset_evaluate(transition_zerofier, self.generator, self.omega, self.fri.domain_length)
        transition_zerofier_root = Merkle.commit(transition_zerofier_codeword)
        return transition_zerofier, transition_zerofier_codeword, transition_zerofier_root

    def transition_degree_bounds( self, transition_constraints ):
        point_degrees = [1] + [self.original_trace_length+self.num_randomizers-1] * 2*self.num_registers
        return [max( sum(r*l for r, l in zip(point_degrees, k)) for k, v in a.dictionary.items()) for a in transition_constraints]

    def transition_quotient_degree_bounds( self, transition_constraints ):
        return [d - (self.original_trace_length-1) for d in self.transition_degree_bounds(transition_constraints)]

    def max_degree( self, transition_constraints ):
        md = max(self.transition_quotient_degree_bounds(transition_constraints))
        return (1 << (len(bin(md)[2:]))) - 1

    def boundary_zerofiers( self, boundary ):
        zerofiers = []
        for s in range(self.num_registers):
            points = [self.omicron^c for c, r, v in boundary if r == s]
            zerofiers = zerofiers + [Polynomial.zerofier_domain(points)]
        return zerofiers

    def boundary_interpolants( self, boundary ):
        interpolants = []
        for s in range(self.num_registers):
            points = [(c,v) for c, r, v in boundary if r == s]
            domain = [self.omicron^c for c,v in points]
            values = [v for c,v in points]
            interpolants = interpolants + [Polynomial.interpolate_domain(domain, values)]
        return interpolants

    def boundary_quotient_degree_bounds( self, randomized_trace_length, boundary ):
        randomized_trace_degree = randomized_trace_length - 1
        return [randomized_trace_degree - bz.degree() for bz in self.boundary_zerofiers(boundary)]

    def sample_weights( self, number, randomness ):
        return [self.field.sample(blake2b(randomness + bytes(i)).digest()) for i in range(0, number)]

    def prove( self, trace, transition_constraints, boundary, transition_zerofier, transition_zerofier_codeword, proof_stream=None ):
        # create proof stream object if necessary
        if proof_stream == None:
            proof_stream = ProofStream()
        
        # concatenate randomizers
        for k in range(self.num_randomizers):
            trace = trace + [[self.field.sample(os.urandom(17)) for s in range(self.num_registers)]]

        # interpolate
        trace_domain = [self.omicron^i for i in range(len(trace))]
        trace_polynomials = []
        for s in range(self.num_registers):
            single_trace = [trace[c][s] for c in range(len(trace))]
            trace_polynomials = trace_polynomials + [fast_interpolate(trace_domain, single_trace, self.omicron, self.omicron_domain_length)]

        # subtract boundary interpolants and divide out boundary zerofiers
        boundary_quotients = []
        for s in range(self.num_registers):
            interpolant = self.boundary_interpolants(boundary)[s]
            zerofier = self.boundary_zerofiers(boundary)[s]
            quotient = (trace_polynomials[s] - interpolant) / zerofier
            boundary_quotients += [quotient]

        # commit to boundary quotients
        boundary_quotient_codewords = []
        boundary_quotient_Merkle_roots = []
        for s in range(self.num_registers):
            boundary_quotient_codewords = boundary_quotient_codewords + [fast_coset_evaluate(boundary_quotients[s], self.generator, self.omega, self.fri_domain_length)]
            merkle_root = Merkle.commit(boundary_quotient_codewords[s])
            proof_stream.push(merkle_root)

        # symbolically evaluate transition constraints
        point = [Polynomial([self.field.zero(), self.field.one()])] + trace_polynomials + [tp.scale(self.omicron) for tp in trace_polynomials]
        transition_polynomials = [a.evaluate_symbolic(point) for a in transition_constraints]

        # divide out zerofier
        transition_quotients = [fast_coset_divide(tp, transition_zerofier, self.generator, self.omicron, self.omicron_domain_length) for tp in transition_polynomials]

        # commit to randomizer polynomial
        randomizer_polynomial = Polynomial([self.field.sample(os.urandom(17)) for i in range(self.max_degree(transition_constraints)+1)])
        randomizer_codeword = fast_coset_evaluate(randomizer_polynomial, self.generator, self.omega, self.fri_domain_length)
        randomizer_root = Merkle.commit(randomizer_codeword)
        proof_stream.push(randomizer_root)

        # get weights for nonlinear combination
        #  - 1 randomizer
        #  - 2 for every transition quotient
        #  - 2 for every boundary quotient
        weights = self.sample_weights(1 + 2*len(transition_quotients) + 2*len(boundary_quotients), proof_stream.prover_fiat_shamir())

        assert([tq.degree() for tq in transition_quotients] == self.transition_quotient_degree_bounds(transition_constraints)), "transition quotient degrees do not match with expectation"

        # compute terms of nonlinear combination polynomial
        x = Polynomial([self.field.zero(), self.field.one()])
        max_degree = self.max_degree(transition_constraints)
        terms = []
        terms += [randomizer_polynomial]
        for i in range(len(transition_quotients)):
            terms += [transition_quotients[i]]
            shift = max_degree - self.transition_quotient_degree_bounds(transition_constraints)[i]
            terms += [(x^shift) * transition_quotients[i]]
        for i in range(self.num_registers):
            terms += [boundary_quotients[i]]
            shift = max_degree - self.boundary_quotient_degree_bounds(len(trace), boundary)[i]
            terms += [(x^shift) * boundary_quotients[i]]

        # take weighted sum
        # combination = sum(weights[i] * terms[i] for all i)
        combination = reduce(lambda a, b : a+b, [Polynomial([weights[i]]) * terms[i] for i in range(len(terms))], Polynomial([]))

        # compute matching codeword
        combined_codeword = fast_coset_evaluate(combination, self.generator, self.omega, self.fri_domain_length)

        # prove low degree of combination polynomial, and collect indices
        indices = self.fri.prove(combined_codeword, proof_stream)

        # process indices
        duplicated_indices = [i for i in indices] + [(i + self.expansion_factor) % self.fri.domain_length for i in indices]
        quadrupled_indices = [i for i in duplicated_indices] + [(i + (self.fri.domain_length // 2)) % self.fri.domain_length for i in duplicated_indices]
        quadrupled_indices.sort()

        # open indicated positions in the boundary quotient codewords
        for bqc in boundary_quotient_codewords:
            for i in quadrupled_indices:
                proof_stream.push(bqc[i])
                path = Merkle.open(i, bqc)
                proof_stream.push(path)

        # ... as well as in the randomizer
        for i in quadrupled_indices:
            proof_stream.push(randomizer_codeword[i])
            path = Merkle.open(i, randomizer_codeword)
            proof_stream.push(path)

        # ... and also in the zerofier!
        for i in quadrupled_indices:
            proof_stream.push(transition_zerofier_codeword[i])
            path = Merkle.open(i, transition_zerofier_codeword)
            proof_stream.push(path)

        # the final proof is just the serialized stream
        return proof_stream.serialize()

    def verify( self, proof, transition_constraints, boundary, transition_zerofier_root, proof_stream=None ):
        H = blake2b

        # infer trace length from boundary conditions
        original_trace_length = 1 + max(c for c, r, v in boundary)
        randomized_trace_length = original_trace_length + self.num_randomizers

        # deserialize with right proof stream
        if proof_stream == None:
            proof_stream = ProofStream()
        proof_stream = proof_stream.deserialize(proof)

        # get Merkle roots of boundary quotient codewords
        boundary_quotient_roots = []
        for s in range(self.num_registers):
            boundary_quotient_roots = boundary_quotient_roots + [proof_stream.pull()]

        # get Merkle root of randomizer polynomial
        randomizer_root = proof_stream.pull()

        # get weights for nonlinear combination
        weights = self.sample_weights(1 + 2*len(transition_constraints) + 2*len(self.boundary_interpolants(boundary)), proof_stream.verifier_fiat_shamir())

        # verify low degree of combination polynomial
        polynomial_values = []
        verifier_accepts = self.fri.verify(proof_stream, polynomial_values)
        polynomial_values.sort(key=lambda iv : iv[0])
        if not verifier_accepts:
            return False

        indices = [i for i,v in polynomial_values]
        values = [v for i,v in polynomial_values]

        # read and verify leafs, which are elements of boundary quotient codewords
        duplicated_indices = [i for i in indices] + [(i + self.expansion_factor) % self.fri.domain_length for i in indices]
        duplicated_indices.sort()
        leafs = []
        for r in range(len(boundary_quotient_roots)):
            leafs = leafs + [dict()]
            for i in duplicated_indices:
                leafs[r][i] = proof_stream.pull()
                path = proof_stream.pull()
                verifier_accepts = verifier_accepts and Merkle.verify(boundary_quotient_roots[r], i, path, leafs[r][i])
                if not verifier_accepts:
                    return False

        # read and verify randomizer leafs
        randomizer = dict()
        for i in duplicated_indices:
            randomizer[i] = proof_stream.pull()
            path = proof_stream.pull()
            verifier_accepts = verifier_accepts and Merkle.verify(randomizer_root, i, path, randomizer[i])
            if not verifier_accepts:
                return False

        # read and verify transition zerofier leafs
        transition_zerofier = dict()
        for i in duplicated_indices:
            transition_zerofier[i] = proof_stream.pull()
            path = proof_stream.pull()
            verifier_accepts = verifier_accepts and Merkle.verify(transition_zerofier_root, i, path, transition_zerofier[i])
            if not verifier_accepts:
                return False

        # verify leafs of combination polynomial
        for i in range(len(indices)):
            current_index = indices[i] # do need i

            # get trace values by applying a correction to the boundary quotient values (which are the leafs)
            domain_current_index = self.generator * (self.omega^current_index)
            next_index = (current_index + self.expansion_factor) % self.fri.domain_length
            domain_next_index = self.generator * (self.omega^next_index)
            current_trace = [self.field.zero() for s in range(self.num_registers)]
            next_trace = [self.field.zero() for s in range(self.num_registers)]
            for s in range(self.num_registers):
                zerofier = self.boundary_zerofiers(boundary)[s]
                interpolant = self.boundary_interpolants(boundary)[s]

                current_trace[s] = leafs[s][current_index] * zerofier.evaluate(domain_current_index) + interpolant.evaluate(domain_current_index)
                next_trace[s] = leafs[s][next_index] * zerofier.evaluate(domain_next_index) + interpolant.evaluate(domain_next_index)

            point = [domain_current_index] + current_trace + next_trace
            transition_constraints_values = [transition_constraints[s].evaluate(point) for s in range(len(transition_constraints))]

            # compute nonlinear combination
            counter = 0
            terms = []
            terms += [randomizer[current_index]]
            for s in range(len(transition_constraints_values)):
                tcv = transition_constraints_values[s]
                quotient = tcv / transition_zerofier[current_index]
                terms += [quotient]
                shift = self.max_degree(transition_constraints) - self.transition_quotient_degree_bounds(transition_constraints)[s]
                terms += [quotient * (domain_current_index^shift)]
            for s in range(self.num_registers):
                bqv = leafs[s][current_index] # boundary quotient value
                terms += [bqv]
                shift = self.max_degree(transition_constraints) - self.boundary_quotient_degree_bounds(randomized_trace_length, boundary)[s]
                terms += [bqv * (domain_current_index^shift)]
            combination = reduce(lambda a, b : a+b, [terms[j] * weights[j] for j in range(len(terms))], self.field.zero())

            # verify against combination polynomial value
            verifier_accepts = verifier_accepts and (combination == values[i])
            if not verifier_accepts:
                return False

        return verifier_accepts

