from fri import *
from univariate import *
from multivariate import *
from functools import reduce
import os

class Stark:
    def __init__( self, field, expansion_factor, num_colinearity_checks, security_level, state_width, num_cycles, air_blowup_factor=2 ):
        assert(len(bin(field.p)) - 2 >= security_level), "p must have at least as many bits as security level"
        assert(expansion_factor & (expansion_factor - 1) == 0), "expansion factor must be a power of 2"
        assert(expansion_factor >= 4), "expansion factor must be 4 or greater"
        assert(num_colinearity_checks * 2 >= security_level), "number of colinearity checks must be at least half of security level"

        self.field = field
        self.expansion_factor = expansion_factor
        self.num_colinearity_checks = num_colinearity_checks
        self.security_level = security_level

        self.num_randomizers = 4*num_colinearity_checks + 1

        self.state_width = state_width
        self.trace_length = num_cycles
        
        randomized_trace_length = self.trace_length + self.num_randomizers
        self.omicron_domain_length = 1 << len(bin(randomized_trace_length * air_blowup_factor)[2:])
        fri_domain_length = self.omicron_domain_length * expansion_factor

        self.generator = self.field.generator()
        self.omega = self.field.primitive_nth_root(fri_domain_length)
        self.omicron = self.field.primitive_nth_root(self.omicron_domain_length)
        self.omicron_domain = [self.omicron^i for i in range(self.omicron_domain_length)]

        self.fri = Fri(self.generator, self.omega, fri_domain_length, self.expansion_factor, self.num_colinearity_checks)

    def composition_degree_bounds( self, air ):
        point_degrees = [1] + [self.trace_length+self.num_randomizers-1] * 2*self.state_width
        return [max( sum(r*l for r, l in zip(point_degrees, k)) for k, v in a.dictionary.items()) for a in air]

    def transition_quotient_degree_bounds( self, air ):
        return [d - (self.trace_length-1) for d in self.composition_degree_bounds(air)]

    def max_degree( self, air ):
        md = max(self.transition_quotient_degree_bounds(air))
        return (1 << (len(bin(md)[2:]))) - 1

    def transition_zerofier( self ):
        domain = self.omicron_domain[0:(self.trace_length-1)]
        return Polynomial.zerofier_domain(domain)

    def boundary_zerofiers( self, boundary ):
        zerofiers = []
        for s in range(self.state_width):
            points = [self.omicron^c for c, r, v in boundary if r == s]
            zerofiers = zerofiers + [Polynomial.zerofier_domain(points)]
        return zerofiers

    def boundary_interpolants( self, boundary ):
        interpolants = []
        for s in range(self.state_width):
            points = [(c,v) for c, r, v in boundary if r == s]
            domain = [self.omicron^c for c,v in points]
            values = [v for c,v in points]
            interpolants = interpolants + [Polynomial.interpolate_domain(domain, values)]
        return interpolants

    def sample_weights( self, number, randomness ):
        return [self.field.sample(blake2b(randomness + bytes(i)).digest()) for i in range(0, number)]

    def prove( self, trace, air, boundary ):
        # create proof stream object
        proof_stream = ProofStream()
        
        # concatenate randomizers
        for k in range(self.num_randomizers):
            trace = trace + [[self.field.sample(os.urandom(17)) for s in range(self.state_width)]]

        # interpolate
        omicron_domain = [self.omicron^i for i in range(self.omicron_domain_length)]
        trace_domain = [self.omicron^i for i in range(len(trace))]
        trace_polynomials = []
        for s in range(self.state_width):
            single_trace = [trace[c][s] for c in range(len(trace))]
            trace_polynomials = trace_polynomials + [Polynomial.interpolate_domain(trace_domain, single_trace)]

        # divide out boundary interpolants
        dense_trace_polynomials = []
        for s in range(self.state_width):
            interpolant = self.boundary_interpolants(boundary)[s]
            zerofier = self.boundary_zerofiers(boundary)[s]
            quotient = (trace_polynomials[s] - interpolant) / zerofier
            dense_trace_polynomials += [quotient]

        # commit to dense trace polynomials
        fri_domain = self.fri.eval_domain()
        dense_trace_codewords = []
        dense_trace_Merkle_roots = []
        for s in range(self.state_width):
            dense_trace_codewords = dense_trace_codewords + [dense_trace_polynomials[s].evaluate_domain(fri_domain)]
            merkle_root = Merkle.commit(dense_trace_codewords[s])
            proof_stream.push(merkle_root)

        # apply air polynomials
        # funny story: Alan had a hard time finding the first polynomial in this list
        point = [Polynomial([self.field.zero(), self.field.one()])] + trace_polynomials + [tp.scale(self.omicron) for tp in trace_polynomials]
        transition_polynomials = [a.evaluate_symbolic(point) for a in air]

        # divide out zerofier
        transition_quotients = [tp / self.transition_zerofier() for tp in transition_polynomials]

        # commit to randomizer polynomial
        randomizer_polynomial = Polynomial([self.field.sample(os.urandom(17)) for i in range(self.max_degree(air)+1)])
        randomizer_codeword = randomizer_polynomial.evaluate_domain(fri_domain) 
        randomizer_root = Merkle.commit(randomizer_codeword)
        proof_stream.push(randomizer_root)

        # get weights for linear combination
        weights = self.sample_weights(1+2*len(transition_quotients), proof_stream.prover_fiat_shamir())

        assert([tq.degree() for tq in transition_quotients] == self.transition_quotient_degree_bounds(air)), "transition quotient degrees do not match with expectation"

        # compute composition polynomial
        x = Polynomial([self.field.zero(), self.field.one()])
        terms = []
        for i in range(len(transition_quotients)):
            terms += [transition_quotients[i]]
            shift = self.max_degree(air) - self.transition_quotient_degree_bounds(air)[i]
            terms += [(x^shift) * transition_quotients[i]]
        terms += [randomizer_polynomial]

        # take weighted sum
        # composition = sum(weights[i] * terms[i] for all i)
        composition = reduce(lambda a, b : a+b, [Polynomial([weights[i]]) * terms[i] for i in range(len(terms))], Polynomial([]))

        # compute codeword
        composition_codeword = composition.evaluate_domain(fri_domain)

        # prove low degree of composition polynomial
        indices = self.fri.prove(composition_codeword, proof_stream)
        indices.sort()
        duplicated_indices = [i for i in indices] + [(i + self.expansion_factor) % self.fri.domain_length for i in indices]

        # open indicated positions in the commitment to the trace
        for dtc in dense_trace_codewords:
            for i in duplicated_indices:
                proof_stream.push(dtc[i])
                path = Merkle.open(i, dtc)
                proof_stream.push(path)

        # .. as well as in the randomizer
        for i in indices:
            proof_stream.push(randomizer_codeword[i])
            path = Merkle.open(i, randomizer_codeword)
            proof_stream.push(path)

        # the final proof is just the serialized stream
        return proof_stream.serialize()

    def verify( self, proof, air, boundary ):
        H = blake2b

        # deserialize
        proof_stream = ProofStream.deserialize(proof)

        # get Merkle roots of dense trace codewords
        dense_trace_roots = []
        for s in range(self.state_width):
            dense_trace_roots = dense_trace_roots + [proof_stream.pull()]

        # get Merkle root of randomizer polynomial
        randomizer_root = proof_stream.pull()

        # get weights for linear combination
        weights = self.sample_weights(1+2*len(air), proof_stream.verifier_fiat_shamir())

        # verify low degree of composition polynomial
        polynomial_values = []
        verifier_accepts = self.fri.verify(proof_stream, polynomial_values)
        polynomial_values.sort(key=lambda iv : iv[0])
        if not verifier_accepts:
            return False

        indices = [i for i,v in polynomial_values]
        values = [v for i,v in polynomial_values]

        # read and verify (dense) trace leafs
        duplicated_indices = [i for i in indices] + [(i + self.expansion_factor) % self.fri.domain_length for i in indices]
        leafs = []
        for r in range(len(dense_trace_roots)):
            leafs = leafs + [dict()]
            for i in duplicated_indices:
                leafs[r][i] = proof_stream.pull()
                path = proof_stream.pull()
                verifier_accepts = verifier_accepts and Merkle.verify(dense_trace_roots[r], i, path, leafs[r][i])
                if not verifier_accepts:
                    return False

        # read and verify randomizer leafs
        randomizer = dict()
        for i in indices:
            randomizer[i] = proof_stream.pull()
            path = proof_stream.pull()
            verifier_accepts = verifier_accepts and Merkle.verify(randomizer_root, i, path, randomizer[i])

        # verify leafs of composition polynomial
        for i in range(len(indices)):
            index = indices[i] # do need i

            # get trace values by applying a correction to dense trace values
            domain_current_index = self.generator * (self.omega^index)
            next_index = (index + self.expansion_factor) % self.fri.domain_length
            domain_next_index = self.generator * (self.omega^next_index)
            previous_trace = [self.field.zero() for s in range(self.state_width)]
            next_trace = [self.field.zero() for s in range(self.state_width)]
            for s in range(self.state_width):
                zerofier = self.boundary_zerofiers(boundary)[s]
                interpolant = self.boundary_interpolants(boundary)[s]

                previous_trace[s] = leafs[s][index] * zerofier.evaluate(domain_current_index) + interpolant.evaluate(domain_current_index)
                next_trace[s] = leafs[s][next_index] * zerofier.evaluate(domain_next_index) + interpolant.evaluate(domain_next_index)

            point = [domain_current_index] + previous_trace + next_trace
            air_values = [air[s].evaluate(point) for s in range(len(air))]

            # compute linear combination
            counter = 0
            terms = []
            for s in range(len(air_values)):
                av = air_values[s]
                quotient = av / self.transition_zerofier().evaluate(domain_current_index)
                terms += [quotient]
                shift = self.max_degree(air) - self.transition_quotient_degree_bounds(air)[s]
                terms += [quotient * (domain_current_index^shift)]
            terms += [randomizer[index]]
            licombo = reduce(lambda a, b : a+b, [terms[j] * weights[j] for j in range(len(terms))], self.field.zero())

            # verify against composition polynomial value
            verifier_accepts = verifier_accepts and (licombo == values[i])
            if not verifier_accepts:
                return False

        return verifier_accepts

