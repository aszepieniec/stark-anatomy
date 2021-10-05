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
