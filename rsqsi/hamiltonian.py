'''
hamiltonian.py -- perturbative problem specification 


The general approach:
    1. Separate complicated ops into their diagonal and off diagonal supports
    2. Decompose off-diagonal parts into strings

'''

import sympy
from dataclasses import dataclass
import lattice
import numpy as np



