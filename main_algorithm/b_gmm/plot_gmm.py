"""
Plot the GMM plot like Fig. 2 b) in our algorithm paper EDL submission.

Developed by Marcel Robitaille on 2022/03/18 Copyright © 2021 QuIN Lab
"""

from operator import sub

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator

from mode import Mode
from figure_constants import FILTERED, GMM, TICK_PADDING, LABEL_FONT_SIZE


def gaussian_mixture(x, base, sep, p, sig):
    """
    Plot the mixture of 2 Gaussian functions with given separation
    """

    g1 = p / (np.sqrt(2 * np.pi) * sig) * \
        np.exp(-(x - base) ** 2 / (2 * sig ** 2))
    g2 = (1 - p) / (np.sqrt(2 * np.pi) * sig) * \
        np.exp(-(x - (base + sep)) ** 2 / (2 * sig ** 2))
    return g1 + g2


def calculate_separation_errors(theory, fitted):
    """
    Associate fitting results to theoretical results
    Find the separation that is the closest to a theoretical one, then
    remove each from consideration
    """
    # pylint: disable=cell-var-from-loop

    theory = theory.copy()
    fitted = fitted.copy()
    while fitted and theory:
        closest = {
            f: min(theory, key=lambda t: abs(f - t))
            for f in fitted
        }
        f, t = min(closest.items(), key=lambda args: abs(sub(*args)))
        yield t, f, abs((f - t) / t)
        theory.remove(t)
        fitted.remove(f)

    for g in fitted:
        yield None, g, None

    for t in theory:
        yield t, None, None


def group_near_values(values, clump_size=3):
    """
    Group values closer than clump_size
    """

    groups = {}
    for i, x in enumerate(values):
        groups[i] = [x]
        for j in groups.keys():  # pylint: disable=consider-using-dict-items
            if i == j:
                continue
            if abs(x - values[j]) < clump_size:
                groups[i] = j
                while not isinstance(groups[j], list):
                    j = groups[j]
                groups[j].append(x)
    groups = [(x, len(x)) for x in groups.values() if isinstance(x, list)]

    return groups


def plot_gmm(example, fit, n_traps, suffix='', seeds=None, output_format='pdf'):
    """
    Make the GMM plot (Fig. 2 (b))
    """

    df = example.read()

    # We don't have labels for real data
    theory = example.parameters.read()['amplitude'].tolist() \
        if 'rtn_sum' in df.columns else None 

    kde_data = example.kde_data.read().squeeze()
    dens = np.array(kde_data.density)
    bins = np.array(kde_data.intensity)

    # print('Theory', theory)
    # traps = [fit.params[f'sep_trap{i}'].value for i in range(n_traps)]
    # print('coupling_factor', fit.params['coupling_factor'].value)
    # print('Traps', traps)
    # errors = pd.DataFrame(
    #     calculate_separation_errors(theory, traps),
    #     columns=['theory', 'fitted', 'sep_error'],
    # )

    # errors['traps'] = len(theory)
    # print(errors)
    # example.sep_error.write(errors)

    # errors['sep_error'] = errors['sep_error'] \
    #     .apply(lambda e: '{:.4f}%'.format(100 * e))

    fig = plt.figure(figsize=(5, 2.40))

    params = fit.params.valuesdict()

    centers = [
        fit.params[f'g{i:0{n_traps}b}_center'].value
        for i in range(2 ** n_traps)
    ]
    ax = fig.add_subplot()
    ax.plot(bins, dens, label='KDE', c=FILTERED, linestyle='solid', zorder=8)
    ax.plot(bins, fit.best_fit, label='GMM', c=GMM, linestyle='dashdot', zorder=9)
    lims = np.where((dens + fit.best_fit) > 1e-4)[0]
    if len(lims):
        ax.set_xlim(bins[lims[0]], bins[lims[-1]])
    ax.set_ylim(0, max(dens) * 2)

    print('centers', centers)
    PUBLICATION = False
    if not PUBLICATION:
        for x in centers:
            ax.axvline(
                x,
                ymin=0.33,
                ymax=0.66,
                linestyle='dashed',
                linewidth=0.5,
                alpha=0.3,
                color='k',
            )

        if seeds:
            seeds_groups = group_near_values(seeds)
            for seed in seeds:
                count = next(count for group, count in seeds_groups
                             if seed in group)
                ax.axvline(
                    seed,
                    ymin=0.0,
                    ymax=0.33,
                    linestyle='dashdot',
                    linewidth=0.5,
                    alpha=0.5,
                    color=(['k', 'tab:orange'] + ['r'] * 100)[count - 1],
                )

    if theory is not None:
        # Group similar values and give vertical lines different colour
        # Makes it easier to identify overlaps
        truth_peaks = list(set(df.rtn_sum) | set(theory))
        print('truth_peaks', truth_peaks)
        truth_peaks = np.array([sum(
            amp
            for j, amp in enumerate(theory)
            if i & 1 << j
        ) for i in range(2 ** n_traps)])
        truth_peaks_groups = group_near_values(truth_peaks)

        for x in truth_peaks:
            count = next(count for group, count in truth_peaks_groups
                         if x in group)
            ax.axvline(x, ymin=0 if PUBLICATION else 0.66, ymax=1,
                       linestyle='dashdot',
                       linewidth=0.5,
                       alpha=0.5,
                       color=(['k', 'tab:orange'] + ['r'] * 100)[count - 1],
                       )
    ax.set_ylabel(
        r'Density (arb. unit\textsuperscript{-1})',
        labelpad=6,
        fontsize=LABEL_FONT_SIZE,
    )
    ax.set_xlabel('Intensity (arb. unit)', labelpad=0, fontsize=LABEL_FONT_SIZE)

    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    ax.tick_params(axis='both', which='both', direction='in', top=True,
                   right=True)
    if PUBLICATION:
        yticks = np.arange(0, 0.05, 0.01)
        ax.set_yticks(yticks)
        # TODO: Figure out why this was commented
        # ax.set_yticklabels(
        #     map('{:.2f}'.format, yticks),
        #     rotation=90,
        #     va='center',
        # )
    ax.tick_params(axis='y', pad=TICK_PADDING)
    # TODO: Figure out what this does an why it was commented
    mf = ticker.ScalarFormatter(useMathText=True)
    mf.set_powerlimits((-2, 2))
    # ax.yaxis.set_major_formatter(mf)

    components = [gaussian_mixture(
        bins,
        params['base'],
        params[f'sep_trap{i}'],
        params[f'p_trap{i}'],
        params['sig'] / 2,
    ) for i in range(n_traps)]
    inset = ax.inset_axes([0.4, 0.5, 0.575, 0.45])
    inset.set_ylabel('Density', labelpad=1, fontsize=10)
    inset.set_xlabel('Intensity', labelpad=0, fontsize=10)

    ax.legend(loc='upper left', prop=dict(size=10), facecolor='white',
              edgecolor='white', framealpha=1, frameon=True, borderaxespad=0.3,
              handletextpad=0.3, labelspacing=0.3)
    for i, (component, colour, linestyle) in enumerate(zip(
            components,
            ['crimson', 'orange', 'dodgerblue'],
            ['dashdot', 'solid', 'dashed'],
    )):
        inset.plot(
            bins,
            component,
            linestyle=linestyle,
            label=f'Trap {i+1}',
            c=colour,
        )
    inset.xaxis.set_minor_locator(AutoMinorLocator(10))
    inset.yaxis.set_minor_locator(AutoMinorLocator(10))
    inset.tick_params(axis='both', which='both', direction='in',
                      top=True, right=True, labelsize=10, pad=2)
    if PUBLICATION:
        inset.set_yticks([0, 0.1, 0.2, 0.3])
        inset.set_yticklabels(['0.0', '0.1', '0.2', '0.3'],
                              rotation=90, va='center')
    inset.legend(
        prop={'size': 10},
        loc='upper right',
        # handlelength=0.7,
        labelspacing=0.3,
        handletextpad=0.3,
        borderaxespad=0.1,
    )

    lims = np.where(sum(components) > 1e-6)[0]
    inset.set_ylim(0, np.array(components).max().max() * 1.2)
    # for ax in axes:
    if len(lims):
        inset.set_xlim(bins[lims[0]], bins[lims[-1]])

    ax.add_patch(
        Rectangle((90, 0.0165), 130, 0.02, facecolor='white', zorder=2))

    fig.tight_layout()
    fig.subplots_adjust(wspace=0)
    figure_filename = \
            example.with_name(f'decomposition{suffix}.{output_format}').path
    fig.savefig(figure_filename, dpi=300, transparent=True)
    plt.close(fig)

    return figure_filename
