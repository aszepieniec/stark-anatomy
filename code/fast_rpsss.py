from rescue_prime import *
from fast_stark import *
from hashlib import blake2s
import os
import pickle as pickle

class SignatureProofStream(ProofStream):
    def __init__( self, document ):
        ProofStream.__init__(self)
        self.document = document
        self.prefix = blake2s(bytes(document)).digest()

    def prover_fiat_shamir( self, num_bytes=32 ):
        return shake_256(self.prefix + self.serialize()).digest(num_bytes)

    def verifier_fiat_shamir( self, num_bytes=32 ):
        return shake_256(self.prefix + pickle.dumps(self.objects[:self.read_index])).digest(num_bytes)

    def deserialize( self, bb ):
        sps = SignatureProofStream(self.document)
        sps.objects = pickle.loads(bb)
        return sps

class FastRPSSS:
    def __init__( self ):
        self.field = Field.main()
        expansion_factor = 4
        num_colinearity_checks = 64
        security_level = 2 * num_colinearity_checks

        self.rp = RescuePrime()
        num_cycles = self.rp.N+1
        state_width = self.rp.m

        self.stark = FastStark(self.field, expansion_factor, num_colinearity_checks, security_level, state_width, num_cycles, transition_constraints_degree=3)
        self.transition_zerofier, self.transition_zerofier_codeword, self.transition_zerofier_root = self.stark.preprocess()

    def stark_prove( self, input_element, proof_stream ):
        output_element = self.rp.hash(input_element)

        trace = self.rp.trace(input_element)
        transition_constraints = self.rp.transition_constraints(self.stark.omicron)
        boundary_constraints = self.rp.boundary_constraints(output_element)
        proof = self.stark.prove(trace, transition_constraints, boundary_constraints, self.transition_zerofier, self.transition_zerofier_codeword, proof_stream)
 
        return proof

    def stark_verify( self, output_element, stark_proof, proof_stream ):
        boundary_constraints = self.rp.boundary_constraints(output_element)
        transition_constraints = self.rp.transition_constraints(self.stark.omicron)
        return self.stark.verify(stark_proof, transition_constraints, boundary_constraints, self.transition_zerofier_root, proof_stream)

    def keygen( self ):
        sk = self.field.sample(os.urandom(17))
        pk = self.rp.hash(sk)
        return sk, pk

    def sign( self, sk, document ):
        sps = SignatureProofStream(document)
        signature = self.stark_prove(sk, sps)
        return signature

    def verify( self, pk, document, signature ):
        sps = SignatureProofStream(document)
        return self.stark_verify(pk, signature, sps)

