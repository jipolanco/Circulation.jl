#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import h5py
from collections import OrderedDict
import sys

if len(sys.argv) > 1:
    STATS_FILE = sys.argv[1]
else:
    print('USAGE: plot_S2.py STATS.h5')
    sys.exit(1)

with h5py.File(STATS_FILE, 'r') as ff:
    gbase = ff['/Increments/Velocity/Longitudinal']
    r = gbase['increments'][:]
    n = r.size - 5
    r = r[:n]
    g = gbase['Moments']
    p = g['p_abs'][:]
    assert p[1] == 2  # second-order structure function
    S2 = g['M_abs'][:n, 1]

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(r, S2 / S2.max(), '.-', label='$S_2(r)$')
ax.plot(r[1:-4], 1.6e-2 * r[1:-4], ':k', label='$r^1$')
ax.set_xlabel("$r$")
ax.set_title("Second-order structure function (circulation code)")
ax.legend()

plt.show()
