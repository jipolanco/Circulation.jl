#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import h5py

STATS_FILE = 'tangle_1024.h5'

print('Loading file:', STATS_FILE)

QUANTITIES = (
    'Velocity',
    'RegVelocity',
    'Momentum',
)


def plot_pdf(ax: plt.Axes, g: h5py.Group, params, plot_kw={}):
    rs = g.parent['loop_sizes'][:] / params['nxi']  # r / ξ
    Nr = rs.size
    bins = g['bin_edges'][:] / params['kappa']  # Γ / κ
    x = (bins[:-1] + bins[1:]) / 2
    bin_size = bins[1] - bins[0]  # assume linear bins!

    for r in range(1, Nr, 5):
        Ns = g['total_samples'][r]
        pdf = g['hist'][r, :] / (Ns * bin_size)

        # PDF integral, should be close to 1.
        # It can be a bit smaller, if there are events falling outside of the
        # histogram.
        # print('PDF integral:', pdf.sum() * bin_size)

        ax.plot(x, pdf, label='${:.2f}$'.format(rs[r]),
                **plot_kw)


def plot_moments(ax: plt.Axes, g: h5py.Group, params, logdiff=False,
                 plot_kw={}):
    rs = g.parent['loop_sizes'][:-1]  # we skip the last loop size...
    Mabs = g['M_abs'][:-1, :]  # [Nr, Np]
    ps = g['p_abs'][:]  # moment exponents [Np]
    Np = ps.size

    rs = rs / params['nxi']  # r / ξ
    kappa = params['kappa']
    rl = np.log(rs)

    for i in range(1, Np, 2):
        p = ps[i]
        M = Mabs[:, i]

        if logdiff:
            x = (rs[1:] + rs[:-1]) / 2
            Ml = np.log(M)
            M = (Ml[1:] - Ml[:-1])  / (rl[1:] - rl[:-1])
        else:
            x = rs
            M[:] /= kappa **p

        ax.plot(x, M, label=f'$p = {p}$', **plot_kw)


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
        plot_pdf(ax, g['Histogram'], params)
        ax.set_yscale('log')
        ax.set_title(name)
        ax.set_xlabel('$Γ / κ$')
        if j == 0:
            ax.set_ylabel('Probability')
        if j == 1:
            ax.legend(fontsize='x-small', ncol=1, title='$r / ξ$')

        ax = axes[1, j]
        logdiff = True
        plot_moments(ax, g['Moments'], params, logdiff=logdiff,
                     plot_kw=dict(marker='x'))
        ax.set_xscale('log')
        if logdiff:
            ylab = r'$\mathrm{d} \, \log ⟨ |Γ|^p ⟩ / \mathrm{d} \, \log r$'
        else:
            ax.set_yscale('log')
            ylab = r'$⟨ |Γ|^p ⟩ / κ^p$'
        ax.set_xlabel('$r / ξ$')
        if j == 0:
            ax.set_ylabel(ylab)
        if j == 1:
            ax.legend(fontsize='x-small', ncol=2)


plt.show()
