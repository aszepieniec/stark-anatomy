from ip import *

def test_serialize( ):
    proof1 = ProofStream()
    proof1.push(1)
    proof1.push({1: '1'})
    proof1.push([1])
    proof1.push(2)

    serialized = proof1.serialize()
    proof2 = ProofStream.deserialize(serialized)

    assert(proof1.pull() == proof2.pull()), "pulled object 0 don't match"
    assert(proof1.pull() == proof2.pull()), "pulled object 1 don't match"
    assert(proof1.pull() == proof2.pull()), "pulled object 2 don't match"
    assert(proof1.pull() == 2), "object 3 pulled from proof1 is not 2"
    assert(proof2.pull() == 2), "object 3 pulled from proof2 is not 2"
    assert(proof1.prover_fiat_shamir() == proof2.prover_fiat_shamir()), "fiat shamir is not the same"
