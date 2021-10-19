from rpsss import *
from fast_rpsss import *
from time import time

def test_rpsss( ):
    print("Testing R'*K signature scheme ...")
    rpsss = RPSSS()

    tick = time()
    sk, pk = rpsss.keygen()
    tock = time()
    print("KeyGen:", (tock - tick), "seconds")

    doc = bytes("Hello, world!", "utf-8")
    tick = time()
    sig = rpsss.sign(sk, doc)
    tock = time()
    print("Sign:", (tock - tick), "seconds")

    tick = time()
    valid = rpsss.verify(pk, doc, sig)
    tock = time()
    print("Verify:", (tock - tick), "seconds")

    if valid:
        print("successfully verified correct signature! \\o/")
    else:
        print("correctly generated signature not valid. <O>")

    not_doc = bytes("Byebye.", "utf-8")
    tick = time()
    valid = rpsss.verify(pk, not_doc, sig)
    tock = time()
    print("Verify:", (tock - tick), "seconds")

    if valid:
        print("signature authenticates bad document <O>")
    else:
        print("signature fails to authenticate bad document! \\o/")

    print("size of signature:", len(sig), "bytes, or ", len(sig) / (2**13), "kB")

def test_fast_rpsss( ):
    print("Testing *FAST* R'*K signature scheme ...")
    rpsss = FastRPSSS()

    tick = time()
    sk, pk = rpsss.keygen()
    tock = time()
    print("KeyGen:", (tock - tick), "seconds")

    doc = bytes("Hello, world!", "utf-8")
    tick = time()
    sig = rpsss.sign(sk, doc)
    tock = time()
    print("Sign:", (tock - tick), "seconds")

    tick = time()
    valid = rpsss.verify(pk, doc, sig)
    tock = time()
    print("Verify:", (tock - tick), "seconds")

    if valid:
        print("successfully verified correct signature! \\o/")
    else:
        print("correctly generated signature not valid. <O>")

    not_doc = bytes("Byebye.", "utf-8")
    tick = time()
    valid = rpsss.verify(pk, not_doc, sig)
    tock = time()
    print("Verify:", (tock - tick), "seconds")

    if valid:
        print("signature authenticates bad document <O>")
    else:
        print("signature fails to authenticate bad document! \\o/")

    print("size of signature:", len(sig), "bytes, or ", len(sig) / (2**13), "kB")

