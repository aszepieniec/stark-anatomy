from rpsss import *

def test_rpsss( ):
    print("Testing R'*K signature scheme ...")
    rpsss = RPSSS()
    sk, pk = rpsss.keygen()
    doc = bytes("Hello, world!", "utf-8")
    sig = rpsss.sign(sk, doc)
    valid = rpsss.verify(pk, doc, sig)
    if valid:
        print("successfully verified correct signature! \\o/")
    else:
        print("correctly generated signature not valid. <O>")

    not_doc = bytes("Byebye.", "utf-8")
    valid = rpsss.verify(pk, not_doc, sig)
    if valid:
        print("signature authenticates bad document <O>")
    else:
        print("signature fails to authenticate bad document! \\o/")

    print("size of signature:", len(sig), "bytes, or ", len(sig) / (2**13), "kB")
