import cProfile
import pstats

import pyximport

pyximport.install()

cProfile.runctx("im2col_cython.normal()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()
