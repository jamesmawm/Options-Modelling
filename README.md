Bisect.py
=================
The root-finding algorithm using the bisection method.

BS.py
========
Gets the Black-Scholes implied volatility using a choice of Bisection method, Newton's method or Newton-Secant method.

Requires SciPy and NumPy.

Copula.py
=======
Credit-loss modelling functions.

LRTree.py
=======
Returns a 3-tuple of option price, delta and gamma according to a Leisen-Reimer tree.

Uses Version 1 of the Peizer/Pratt inversion formula (see attached paper).

BS formula values are used in the final-step substitution of the LR Tree.

(What better way to price an option on the very last day of expiry?)
