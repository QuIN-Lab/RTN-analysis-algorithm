"""
This small code snippet tests for a correlation between digitization error and
tau error.

Developed by Marcel Robitaille on 2022/03/25 Copyright © 2021 QuIN Lab
"""


# Some "all metrics" aggregated results file
df = pd.read_csv('all_metrics.csv')

for which in ('tau_low', 'tau_high'):
    fig, ax = plt.subplots()
    for n_traps in np.arange(3) + 1:
        df_sub = df[df.n_traps == n_traps]
        ax.scatter(df_sub.digitization_error, df_sub[which], s=0.75,
                   label=f'$N_\mathrm{{traps}}$={n_traps}')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Digitization error')
    ax.set_ylabel({
        'tau_low': 'Tau low error',
        'tau_high': 'Tau high error',
    }[which])
    fig.savefig(boxplot_out_dir / f'correlation_{which}.png', dpi=300)
    plt.close(fig)
