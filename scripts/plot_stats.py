#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import h5py

STATS_FILE = '../notebooks/stats.h5'

KAPPA = 1.0 / np.pi  # not sure about this!

QUANTITIES = (
    'Velocity',
    'RegVelocity',
    'Momentum',
)


def plot_histogram(ax: plt.Axes, g: h5py.Group, plot_kw={}):
    rs = g.parent['loop_sizes'][()]
    Nr = rs.size
    bins = g['bin_edges'][:]
    x = (bins[:-1] + bins[1:]) / 2
    x /= KAPPA
    # name = g.parent.name.replace('/', '')
    for r in range(Nr):
        hist = g['hist'][r, :]
        ax.plot(x, hist, color=f'C{r}', label=f'$r = {rs[r]}$',
                **plot_kw)


def plot_moments(ax: plt.Axes, g: h5py.Group, plot_kw={}):
    rs = g.parent['loop_sizes'][()]
    Mabs = g['M_abs'][:, :]
    # Modd = g['M_odd'][:, :]
    Np = Mabs.shape[1]
    # Nodd = Modd.shape[1]
    for p in range(1, Np, 2):
        # This normalisation works great for the real velocity.
        # Is there a characteristic velocity equal to sqrt(2)?
        norm = (np.sqrt(2) * 2 * np.pi)**(p + 1)
        # norm = KAPPA**(p + 1)
        ax.plot(rs, Mabs[:, p] / norm, label=f'$p = {p + 1}$', **plot_kw)


with h5py.File(STATS_FILE, 'r') as ff:
    fig, axes = plt.subplots(2, 3, figsize=(12, 6),
                             sharex='row', sharey='row')

    for j, name in enumerate(QUANTITIES):
        g = ff[name]

        ax = axes[0, j]
        plot_histogram(ax, g['Histogram'])
        ax.set_yscale('log')
        ax.legend(fontsize='x-small', ncol=2)
        ax.set_title(name)

        ax = axes[1, j]
        plot_moments(ax, g['Moments'], plot_kw=dict(marker='x'))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('$r$')
        ax.legend()


plt.show()
