""""
100 tickets
40 tienen premio

encontrar la f_masa

"""
import numpy as np
import itertools
import scipy.stats as stats

itertools.combinations(100,3)
np.c
p = 0.4

rta = len([*itertools.combinations(range(40), 2)]) * len([*itertools.combinations(range(60), 1)]) / len([*itertools.combinations(range(100), 3)])

stats.binom.pmf()

P(B|Vota) = P(V|B) . P(B) / P(Vota)

P(B) = 0.2
B(Vota) = P(V|A).P(A) + P(V|B).P(B) + P(V|C).P(C) = 0.65 + 0.82 + 0.50 =
P(V|B) = 0.82

# Ej 5

media = 100
desv = 15

valor_a_exceder = 125

p_5 = 1 - stats.norm(100, 15).cdf(valor_a_exceder)

print(p_5)