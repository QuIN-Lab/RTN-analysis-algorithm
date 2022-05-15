# Results presentation

This folder holds all those files responsible for making figures based on the results of the algorithm.
Files beginning in `overleaf_` typically plot figures intended for publication,
whereas the other files make figures for our own interpretation and debugging.
The `overleaf_` files also typically only generate one figure: the specific example we chose to present in our paper.
The other files search for all the examples and make a plot for each.

- [`artn_heatmap.py`](./artn_heatmap.py): Plot heatmaps for aRTN error
- [`boxplots.py`](./boxplots.py): Plot boxplots of individual metrics or combined boxplots (like Fig. 4) for normal RTN and aRTN (this is an exception to the `overleaf_` rule, this file plots Fig. 4)
- [`internal_matrices.py`](./internal_matrices.py): Plot heatmaps for normal RTN error
- [`overleaf_cnt_data_plots.py`](./overleaf_cnt_data_plots.py): Plot the CNT digitization figure (Fig. 5 (a))
- [`overleaf_rnn_plots.py`](./overleaf_rnn_plots.py): Plot the unfiltered and filtered digitization result from the RNN algorithm. Show the false jumping points with arrows. Formerly Fig. 3 (b) and (c), but these have been cut from the paper.
- [`overleaf_rtn_definition_and_wn_examples.py`](./overleaf_rtn_definition_and_wn_examples.py): Plot three examples from our normal 330 dataset and define RTN parameters on leftmost panel (Fig. 1)
- [`overleaf_rtn_metastable_plot.py`](./overleaf_rtn_metastable_plot.py): Make the small example plot of one metastable signal (Fig. 5 (b))
- [`plot_taus.py`](./plot_taus.py): Plot a histogram of the extracted tau values.

All figure references are based on the version of our paper from the appeal submitted Feb. 7, 2022.

Many of these files are using [vim-ipython-cell](https://github.com/hanschen/vim-ipython-cell), a plugin for my text editor to make it act like Jupyter Notebooks.
The code is separated by `# %%` comments, which denote the cells.
The cells are intended to be executed individually, like in Jupyter Notebooks.
