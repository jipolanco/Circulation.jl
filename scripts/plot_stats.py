#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import h5py
from collections import OrderedDict

STATS_FILE = 'tangle_256.h5'

MOMENTS_FROM_HISTOGRAM = False
MOMENTS_FRACTIONAL = False  # show fractional moments?

print('Loading file:', STATS_FILE)

QUANTITIES = OrderedDict(
    Velocity = {
        'name': 'Velocity',
    },
    RegVelocity = {
        'name': 'Regularised velocity',
    },
    Momentum = {
        'name': 'Momentum',
    },
)


def plot_pdf(ax: plt.Axes, g: h5py.Group, params, moment=0, plot_kw={}):
    rs = g.parent['loop_sizes'][:] / params['nxi']  # r / ξ
    Nr = rs.size
    kappa = params['kappa']
    bins = g['bin_edges'][:] / kappa  # Γ / κ
    x = (bins[:-1] + bins[1:]) / 2
    bin_size = bins[1] - bins[0]  # assume linear bins!
    mins = g['minimum'][:] / kappa
    maxs = g['maximum'][:] / kappa

    for r in range(1, Nr, 5):
        Ns = g['total_samples'][r]
        pdf = g['hist'][r, :] / (Ns * bin_size)

        if mins[r] < bins[0] or maxs[r] > bins[-1]:
            print('WARNING: found Γ outside of histogram. Min/max =',
                  (mins[r], maxs[r]))

        # PDF integral, should be close to 1.
        # It can be a bit smaller, if there are events falling outside of the
        # histogram.
        print('PDF integral:', pdf.sum() * bin_size)

        if moment != 0:
            pdf *= x**moment
            # Remove x = 0.
            # Otherwise I get a pdf = 0 point, which looks bad with log scale.
            n = pdf.size // 2
            assert abs(x[n]) < 1e-8
            pdf[n] = np.nan

        ax.plot(x, pdf, label='${:.2f}$'.format(rs[r]),
                **plot_kw)


def load_moments(g: h5py.Group, odd=False, fractional=False):
    if odd:
        M = np.abs(g['M_odd'][:, :])
        ps = g['p_odd'][:]  # moment exponents [Np]
    else:
        M = g['M_abs'][:, :]  # [Nr, Np]
        ps = g['p_abs'][:]  # moment exponents [Np]
        if fractional and 'M_frac' in g:
            # Add fractional moments if they were computed.
            Mf = g['M_frac'][:, :]  # [Nr, Np_frac]
            pf = g['p_frac'][:]     # [Np_frac]
            M = np.append(Mf, M, axis=1)
            ps = np.append(pf, ps)
    return ps, M


def moments_from_histogram(g: h5py.Group):
    hist = g['hist'][:, :]  # [Nr, Nbins]
    bins = g['bin_edges'][:]
    x = (bins[:-1] + bins[1:]) / 2

    # Make sure that x = 0 is really zero.
    n = x.size // 2
    assert abs(x[n]) < 1e-10
    x[n] = 0

    ps = np.arange(1, 21, step=1, dtype=np.int)
    Nr = hist.shape[0]
    Np = ps.size

    Mabs = np.zeros((Nr, Np))

    for i in range(Np):
        Mabs[:, i] = np.sum(hist * np.abs(x)**ps[i], axis=1)

    return ps, Mabs


def plot_moments(ax: plt.Axes, g: h5py.Group, params, logdiff=False,
                 pskip=1, fractional=False, pmax=100, plot_kw={}):
    if g.name.endswith('Moments'):
        ps, Mabs = load_moments(g, fractional=fractional)
    elif g.name.endswith('Histogram'):
        ps, Mabs = moments_from_histogram(g)

    rs = g.parent['loop_sizes'][:]  # [Nr]
    Mabs = Mabs[:, :]  # [Nr, Np]
    Np = ps.size

    rs = rs / params['nxi']  # r / ξ
    kappa = params['kappa']
    rl = np.log(rs)

    for i in range(0, Np, pskip):
        p = ps[i]
        if p > pmax:
            continue

        M = Mabs[:, i]

        if logdiff:
            x = (rs[1:] + rs[:-1]) / 2
            Ml = np.log(M)
            M = (Ml[1:] - Ml[:-1])  / (rl[1:] - rl[:-1])
        else:
            x = rs
            M[:] /= kappa **p

        ax.plot(x, M, label=f'$p = {p}$', **plot_kw)


def output_filename(filein_h5):
    suffix = '_hist' if MOMENTS_FROM_HISTOGRAM else ''
    return filein_h5.replace('.h5', f'{suffix}.svg')


with h5py.File(STATS_FILE, 'r') as ff:
    g_params = ff['/ParamsGP']
    params = dict(
        kappa=g_params['kappa'][()],
        xi=g_params['xi'][()],
        nxi=g_params['nxi'][()],
    )

    g_circ = ff['/Circulation']

    fig, axes = plt.subplots(3, 3, figsize=(12, 8),
                             sharex='row', sharey='row')

    for j, (key, val) in enumerate(QUANTITIES.items()):
        g = g_circ[key]

        ax = axes[0, j]
        moment = 8
        plot_pdf(ax, g['Histogram'], params, moment=moment)
        ax.set_yscale('log')
        ax.set_title(val['name'])
        ax.set_xlabel('$Γ / κ$')
        if j == 0:
            gamma = r'\left( Γ / κ \right)'
            s = '' if moment == 0 else f'{gamma}^{{{moment}}} \\,'
            ax.set_ylabel(f'${s} P{gamma}$')
        if j == 1:
            ax.legend(fontsize='x-small', ncol=1, title='$r / ξ$')

        for i in (1, 2):
            ax = axes[i, j]
            logdiff = i == 2
            gname = 'Histogram' if MOMENTS_FROM_HISTOGRAM else 'Moments'
            if MOMENTS_FRACTIONAL:
                kwargs = dict(fractional=True, pmax=4, pskip=2)
            else:
                kwargs = dict()
            plot_moments(ax, g[gname], params, logdiff=logdiff,
                         plot_kw=dict(marker='x'), **kwargs)
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


    fname = output_filename(STATS_FILE)
    print('Saving', fname)
    fig.savefig(fname)

plt.show()
