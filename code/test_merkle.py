from merkle import Merkle
from os import urandom

def test_merkle():
    n = 64
    leafs = [urandom(int(urandom(1)[0])) for i in range(n)]
    root = Merkle.commit_(leafs)

    # opening any leaf should work
    for i in range(n):
        path = Merkle.open_(i, leafs)
        assert(Merkle.verify_(root, i, path, leafs[i]))

    # opening non-leafs should not work
    for i in range(n):
        path = Merkle.open_(i, leafs)
        assert(False == Merkle.verify_(root, i, path, urandom(51)))

    # opening wrong leafs should not work
    for i in range(n):
        path = Merkle.open_(i, leafs)
        j = (i + 1 + (int(urandom(1)[0] % (n-1)))) % n
        assert(False == Merkle.verify_(root, i, path, leafs[j]))

    # opening leafs with the wrong index should not work
    for i in range(n):
        path = Merkle.open_(i, leafs)
        j = (i + 1 + (int(urandom(1)[0] % (n-1)))) % n
        assert(False == Merkle.verify_(root, j, path, leafs[i]))

    # opening leafs to a false root should not work
    for i in range(n):
        path = Merkle.open_(i, leafs)
        assert(False == Merkle.verify_(urandom(32), i, path, leafs[i]))

    # opening leafs with even one falsehood in the path should not work
    for i in range(n):
        path = Merkle.open_(i, leafs)
        for j in range(len(path)):
            fake_path = path[0:j] + [urandom(32)] + path[j+1:]
            assert(False == Merkle.verify_(root, i, fake_path, leafs[i]))

    # opening leafs to a different root should not work
    fake_root = Merkle.commit_([urandom(32) for i in range(n)])
    for i in range(n):
        path = Merkle.open_(i, leafs)
        assert(False == Merkle.verify_(fake_root, i, path, leafs[i]))
