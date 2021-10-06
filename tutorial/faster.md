# Anatomy of a STARK, Part 6: Speeding Things Up

The previous part of this tutorial posed the question whether maths-level improvement can reduce the running times of the STARK algorithms. Indeed they can! These computational algebra tricks are folklore and independent of the STARK machinery.

## The Number Theoretic Transform and its Applications

### The Fast Fourier Transform

Let $f(X)$ be a polynomial of degree at most $2^k - 1$ with complex numbers as coefficients. What is the most efficient way to find the list of evaluations $f(X)$ on the $2^k$ complex roots of unity? Specifically, let $\omega = e^{2 \pi i / 2^k}$, then the output of the algorithm should be $(f(\omega^i))_{i=0}^{2^k-1} = (f(1), f(\omega), f(\omega^2), \ldots, f(\omega^{2^k-1}))$.

The naïve solution is to sequentially compute each evaluation individually. A more intelligent solution relies on the observation that $f(\omega^i) = \sum_{j=0}^{2^k-1} \omega^{ij} f_j$ and splitting the even and odd terms gives
$$ f(\omega^i) = \sum_{j=0}^{2^{k-1}-1} \omega^{i(2j)}f_{2j} + \sum_{j=0}^{2^{k-1}-1} \omega^{i(2j+1)} f_{2j+1} \\
 = \sum_{j=0}^{2^{k-1}-1} \omega^{i(2j)}f_{2j} + \omega^i \cdot \sum_{j=0}^{2^{k-1}-1} \omega^{i(2j)} f_{2j+1} \\
 = f_E(\omega^{2i}) + \omega^i \cdot f_O(\omega^{2i}) \enspace , $$
where $f_E(X)$ and $f_O(X)$ are the polynomials whose coefficients are the even coefficients, and odd coefficients respectively, of $f(X)$.

In other words, the evaluation of $f(X)$ at $\omega^i$ can be described in terms of the evaluations of $f_E(X)$ and $f_O(X)$ at $\omega^{2i}$. The same is true for a batch of points $\{\omega^{ij}\}_{j=0}^{2^k-1}$, in which case the values of $f_E(X)$ and $f_O(X)$ on a domain of only half the size are needed: $\{(\omega^{ij})^2\}_{j=0}^{2^k-1} = \{(\omega^{2i})^j\}_{j=0}^{2^{k-1}-1}$. Note that tasks of batch-evaluating $f_E(X)$ and $f_O(X)$ are independent tasks of half the size. This screams divide and conquer! Specifically, the following strategy suggests itself:
 - split coefficient vector into even and odd parts;
 - evaluate $f_E(X)$ on $\{(\omega^{2i})^j\}_{j=0}^{2^{k-1}-1}$ by recursion;
 - evaluate $f_O(X)$ on $\{(\omega^{2i})^j\}_{j=0}^{2^{k-1}-1}$ by recursion;
 - merge the evaluation vectors using the formula $f(\omega^i) = f_E(\omega^{2i}) + \omega^i \cdot f_O(\omega^{2i})$.

Vòilá! That's the fast Fourier transform. The reason why the $2^k$th root of unity is needed is because it guarantees that $\{(\omega^{ij})^2\}_{j=0}^{2^k-1} = \{(\omega^{2i})^j\}_{j=0}^{2^{k-1}-1}$, and so the recursion really is on a domain of half the size. (If you were to use a similar strategy to evaluate $f(X)$ in $\{z^j\}_{j=0}^{2^k-1}$ then the evaluation domain would not shrink with every recursion step.) There are $k$ recursion steps, and at each level there are $2^k$ multiplications and additions, so the complexity of this algorithm is $O(2^k \cdot k)$, or expressed in terms of the length of the coefficient vector $N = 2^k$, $O(N \cdot \log N)$.