# Anatomy of a STARK, Part 4: the STARK Polynomial IOP

This part of the tutorial deals with the information-theoretic backbone of the STARK proof system, which you might call the STARK Polynomial IOP. 
Recall that the compilation pipeline of SNARKs involves intermediate stages, the first two of which are the *arithmetic constraint system* and the *Polynomial IOP*. This tutorial describes the properties of the arithmetic constraint system but a discussion about the *arithmetization* step, which transforms the initial computation into an arithmetic constraint system, is out of scope. However, *interpolation* step, which transforms this arithmetic constraint system into a Polynomial IOP, is discussed at length. The final Polynomial IOP can be compiled into a concrete proof system using the FRI-based compiler described in [[part 3|fri.md]].

## Arithmetic Intermediate Representation (AIR)

The *arithmetic intermediate representation (AIR)* (also, arithmetic *internal* representation) is a way of describing a computation in terms of an execution trace that satisfies a number of constraints induced by the correct evolution of the state. The term *arithmetic* refers to the fact that this execution trace consists of a list of finite field elements (or an array, if more than one register is involved), and that the constraints are expressible as low degree polynomials.

Let's make this more concrete. Let $\mathbb{F}_p$ be the field of definition. Without loss of generality, the computation describes the evolution a *state* of $\mathsf{w}$ registers for $T$ cycles. The *algebraic execution trace (AET)* is the table of $T \times \mathsf{w}$ field elements where every row describes the state of the system at the given point in time, and every column tracks the value of the given register. A *state transition function* $f : \mathbb{F}_p^\mathsf{w} \rightarrow \mathbb{F}_p^\mathsf{w}$ determines the state at the next cycle as a function of the state at the previous cycle. Furthermore, a list of boundary conditions $\mathcal{B} : [\mathbb{Z}_T \times \mathbb{Z}_\mathsf{w} \times \mathbb{F}]$  enforce the correct values of some or all registers at the first cycle, last cycle, or even at arbitrary cycles.

The *computational integrity claim* consists of the state transition function and the boundary conditions. The *witness* to this claim is the algebraic execution trace. The claim is *true* if there is a witness $W \in \mathbb{G}^{T \times \mathsf{w}}$ such that:
 - for every cycle, the state evolves correctly: $\forall i \in \mathbb{Z}_T \, . \, f(W_{[i,:]}) = W_{[i+1,:]}$; and
 - all boundary conditions are satisfied: $\forall (i, w, e) \in \mathcal{B} \, . \, W_{[i,w]} = e$.

The state transition function hides a lot of complexity. For the purpose of STARKs, it needs to be describable as low degree polynomials that are *independent of the cycle*. However, this list of polynomials does not need to compute the next state from the current one; it merely needs to distinguish correct evolutions from incorrect ones. Specifically, the function $f : \mathbb{F}_p^\mathsf{w} \rightarrow \mathbb{F}_p^\mathsf{w}$ is represented by a list of polynomials $\mathbf{p} \in \mathbb{F}_p[X_0, \ldots, X_{\mathsf{w}-1}, Y_0, \ldots, Y_{\mathsf{w}-1}]$ such that $f(\mathbf{x}) = \mathbf{y}$ if and only if $\mathbf{p}(\mathbf{x}, \mathbf{y}) = \mathbf{0}$. Say there are $r$ such state transition verification polynomials. Then the transition constraints become:
 - $\forall i \in \mathbb{Z}_T \, . \, \forall j \in \mathbb{Z}_r \, . \, p_r(W_{[i,0]}, \ldots, W_{[i, \mathsf{w}-1]}, W_{[i+1,0]}, \ldots, W_{[i+1, \mathsf{w}-1]}) = 0$.

This representation admits *non-determinism*, which has the capacity to reduce high degree state transition *computation* polynomials with low degree state transition *verification* polynomials. For example: the state transition function $f : \mathbb{F}_p \rightarrow \mathbb{F}_p$ given by
$$ x \mapsto \left\{ \begin{array}{l}
x^{-1} & \Leftarrow x \neq 0 \\
0 & \Leftarrow x = 0 
\end{array} \right. $$
can be represented as a computation polynomial $f(x) = x^{p-1}$ or as a pair of verification polynomials $\mathbf{p}(x,y) = (x(xy-1), y(xy-1))$. The degree drops from $p-1$ to 3.

Not all lists of $\mathsf{w}$ represent valid states. For instance, some registers may be constrained to bits and thus take only values from $\{0, 1\}$. The state transition function is what guarantees that the next state is well-formed if the current state is. When translating to verification polynomials, these *consistency constraints* are polynomials in the ring $\mathbb{F}_p[X_0, \ldots, X_{\mathsf{w}-1}]$ because they apply to every single row in the AET, as opposed to every consecutive pair of rows. For the sake of simplicity, this tutorial will ignore consistency constraints and pretend as though every $\mathsf{w}$-tuple of field elements represents a valid state.

## Interpolation

The arithmetic constraint system described above already represents the computational integrity claim as a bunch polynomials. Transforming this constraint system into a Polynomial IOP requires extending this representation in terms of polynomials to the witness and its validity. Specifically, we need to represent the conditions for true computational integrity claims in terms of identities of polynomials.

Let $D$ be a list of points referred to from here on out as the *trace evaluation domain*. Typically, $D$ is set to the span of a generator $\omicron$ of a subgroup of order $2^k \geq T+1$. So for the time being set $D = \{\omicron^i | i \in \mathbb{Z}\}$. The Greek letter $\omicron$ ("omicron") indicates that the trace evaluation domain is smaller than the FRI evaluation domain by a factor exactly equal to the expansion factor[^1].

Let $\boldsymbol{t}(X) \in (\mathbb{F}_p[X])^\mathsf{w}$ be a list of $\mathsf{w}$ univariate polynomials that interpolate through $W$ on $D$. Specifically, the *trace polynomial* $t_w(X)$ for register $w$ is the univariate polynomial of lowest degree such that $\forall i \in \{0, \ldots, T\} \, . \, t_w(\omicron^i) = W[i, w]$. The trace polynomials are a representation of the algebraic execution trace in terms of univariate polynomials.

Translating the conditions for true computational integrity claims to the trace polynomials, one gets:
 - all boundary constraints are satisfied: $\forall (i, w, e) \in \mathcal{B} \, . \, t_w(\omicron^i) = e$; and
 - for all cycles, all transition constraints are satisfied: $\forall i \in \mathbb{Z}_T \, . \, \forall j \in \mathbb{Z}_r \, . \, p_j( t_0(\omicron^i), \ldots, t_{\mathsf{w}-1}(\omicron^i), t_0(\omicron^{i+1}), \ldots, t_{\mathsf{w}-1}(\omicron^{i+1})) = 0$.

The last expression looks complicated. However, observe that the left hand side of the equation corresponds to the univariate polynomial $p_j(t_0(X)), \ldots, t_{\mathsf{w}-1}(X), t_0(\omicron \cdot X), \ldots, t_{\mathsf{w}-1}(\omicron \cdot X))$. The entire expression simply says that all $r$ of these *composition polynomials* evaluate to 0 in $\{ \omicron^i | i \in \mathbb{Z}_T\}$.

This observation gives rise to the following high-level Polynomial IOP:
 1. The prover commits to the trace polynomials $\boldsymbol{t}(X)$.
 2. The verifier checks that $t_w(X)$ evaluates to $e$ in $\omicron^i$ for all $(i, w, e) \in \mathcal{B}$.
 3. The prover commits to the composition polynomials $\mathbf{c}(X) = \mathbf{p}(t_0(X)), \ldots, t_{\mathsf{w}-1}(X), t_0(\omicron \cdot X), \ldots, t_{\mathsf{w}-1}(\omicron \cdot X))$.
 4. The verifier checks that $\mathbf{c}(X)$ and $\boldsymbol{t}(X)$ are correctly related by:
   4.1. choosing a random point $z \in \mathbb{F}_p \backslash \{0\}$,
   4.2. querying the values of $\boldsymbol{t}(X)$ in $z$ and $\omicron \cdot z$,
   4.3. evaluating the transition verification polynomials $\mathbf{p}(X_1, \ldots, X_{\mathsf{w}-1}, Y_0, \ldots, Y_{\mathsf{w}-1})$ in these $2\mathsf{w}$ points, and
   4.4 querying the values of $\mathbf{c}(X)$ in $z$,
   4.5 checking that the values obtained in the previous two steps match;
 5. The verifier checks that the composition polynomials $\mathbf{c}(X)$ evaluate to zero in $\{ \omicron^i | i \in \mathbb{Z}_T\}$.

In fact, the commitment of the composition polynomials can be omitted. Instead, the verifier uses the evaluation of $\boldsymbol{t}(X)$ in $z$ and $\omicron \cdot z$ to compute the value of $\mathbf{c}(X)$ in the one point needed to verify that $\mathbf{c}(X)$ evaluates to 0 in $\{ \omicron^i | i \in \mathbb{Z}_T\}$.

There is another layer of redundancy, but it is only apparent after the evaluation checks are unrolled. The FRI compiler simulates an evaluation check by a) subtracting the y-coordinate, b) dividing out the zerofier, which is the minimal polynomial that vanishes at the x-coordinate, and c) proving that the resulting quotient has a bounded degree. This procedure happens twice for the STARK polynomials -- first: applied to the trace polynomials to show satisfaction of the boundary constraints, and second: applied to the composition polynomials to show that the transition constraints are satisfied. We call the resulting lists of quotient polynomials the *boundary quotients* and the *transition quotients* respectively.

The redundancy comes from the fact that the trace polynomials relate to both quotients. It can therefore be eliminated by merging the equations they are involved in. The next diagram illustrates this elimination in the context of the STARK Polynomial IOP workflow. The green box indicates that the polynomials are committed to through the familiar evaluation and Merkle root procedure and are provided as input to FRI.

![Overview of the STARK workflow](graphics/stark-workflow.svg)

At the top of this diagram in red are the objects associated with the arithmetic constraint system, with the constraints written in small caps font to indicate that they are known to the verifier. The prover interpolates the execution trace to obtain the trace polynomials, but it is not necessary to commit to these polynomials. Instead, the prover interpolates the boundary points and subtracts the resulting interpolants from the trace polynomials. This procedure produces the *dense trace polynomials*, for lack of a better name. To obtain the boundary quotients from the dense trace polynomials, the prover divides out the zerofier. Note that the boundary quotients and trace polynomials are equivalent in the following sense: if the verifier knows a value in a given point of one, he can compute the matching value of the other using only public information.

To obtain the composition polynomials, the prover evaluates the transition constraints (recall, these are given as multivariate polynomials) symbolically in the trace polynomials. To get the transition quotients from the composition polynomials, divide out the zerofier. Assume for the time being that the verifier is capable of evaluating this zerofier efficiently. Note that the transition quotients and the trace polynomials are not equivalent -- the verifier cannot necessarily undo the symbolic evaluation. However, this non-equivalence does not matter. What the verifier needs to verify is that the boundary quotients and the transition quotients are linked. Traveling from the boundary quotients to the transition quotients, and performing the indicated arithmetic along the way, establishes this link. The remaining part of the entire computational integrity claim is the bounded degree of the quotient polynomials, and this is exactly what FRI already solves.

The use of the plural on the right hand side is slightly misleading. After the boundary quotients have been committed to by sending their Merkle roots to the verifier, the prover obtains from the verifier random weights with which to compress the transition constraints to a single linear combination. As a result of this compression, there is one transition constraint, one composition polynomial, and one transition quotient. Nevertheless, this compression may be omitted without affecting security; it merely requires more work on the part of both the prover and the verifier.

[^1]: It is worth ensuring that the trace evaluation domain is disjoint from the FRI evaluation domain, for example by using the coset-trick. However, if partially overlapping subgroups are used for both domains, then $\omega^{1 / \rho} = \omicron$ and $\omega$ generates the larger domain whereas $\omicron$ generates the smaller one.