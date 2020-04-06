#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import h5py

STATS_FILE = 'tangle_256.h5'

QUANTITIES = (
    'Velocity',
    'RegVelocity',
    'Momentum',
)


def plot_histogram(ax: plt.Axes, g: h5py.Group, params, plot_kw={}):
    rs = g.parent['loop_sizes'][:] / params['nxi']  # r / ξ
    Nr = rs.size
    bins = g['bin_edges'][:]
    x = (bins[:-1] + bins[1:]) / 2
    x /= params['kappa']
    # name = g.parent.name.replace('/', '')
    for r in range(0, Nr, 4):
        hist = g['hist'][r, :]
        ax.plot(x, hist, color=f'C{r}', label='${:.2f}$'.format(rs[r]),
                **plot_kw)


def plot_moments(ax: plt.Axes, g: h5py.Group, params, plot_kw={}):
    rs = g.parent['loop_sizes'][:]
    Mabs = g['M_abs'][:, :]
    # Modd = g['M_odd'][:, :]
    ps = g['p_abs'][:]  # moment exponents
    Np = ps.size
    # Nodd = Modd.shape[1]
    for i in range(1, Np, 2):
        # This normalisation works great for the real velocity.
        # Is there a characteristic velocity equal to sqrt(2)?
        # norm = (np.sqrt(2) * 2 * np.pi)**(p + 1)
        p = ps[i]
        norm = params['kappa']**p
        ax.plot(rs / params['nxi'], Mabs[:, i] / norm,
                label=f'$p = {p}$', **plot_kw)


with h5py.File(STATS_FILE, 'r') as ff:
    g_params = ff['/ParamsGP']
    params = dict(
        kappa=g_params['kappa'][()],
        xi=g_params['xi'][()],
        nxi=g_params['nxi'][()],
    )

    g_circ = ff['/Circulation']

    fig, axes = plt.subplots(2, 3, figsize=(12, 6),
                             sharex='row', sharey='row')

    for j, name in enumerate(QUANTITIES):
        g = g_circ[name]

        ax = axes[0, j]
        plot_histogram(ax, g['Histogram'], params)
        ax.set_yscale('log')
        ax.legend(fontsize='x-small', ncol=1, title='$r / ξ$')
        ax.set_title(name)

        ax = axes[1, j]
        plot_moments(ax, g['Moments'], params, plot_kw=dict(marker='x'))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('$r / ξ$')
        ax.legend(fontsize='x-small', ncol=2)


plt.show()
