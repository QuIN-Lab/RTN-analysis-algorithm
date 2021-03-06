# RTN Analysis Algorithm

This project is our 3-step RTN and aRTN analysis algorithm
that was the subject of our paper submitted to IEEE EDL.

## Theory

Please refer to our submitted paper for a detailed description of the theory of our algorithm.
A preprint is available [here](https://www.researchsquare.com/article/rs-1694395/v1).

See also:
- [Description of the change in τ error calculation](./docs/tau_error_definition.md)
- [Running modes of the algorithm](./docs/algorithm_modes.md)

## Setup

### Anaconda method

1. Clone the repository and enter the directory
	```bash
	git clone ist-git@git.uwaterloo.ca:QuINLab/Projects/Noise/analysis-algorithm.git
	cd analysis-algorithm
	```

1. Create a new conda environment with pip installed
	```bash
	conda create -n analysis_algorithm pip
	```

1. Activate the new environment
	```bash
	conda use analysis_algorithm
	```

1. Install the requirements
	```bash
	pip install -r requirements.txt
	```

### Description of Data Files

This project uses many different types of files. There are two "design value" files for each example:
files ending in `_signals.feather` and `_parameters.csv`.

There are also many intermediate files and result files.
The reason to have each step save it's results to a file
is to easily run a single part of the algorithm.
For example, when I am working on the GMM step, I don't want to have to rerun the KDE step every time, which may be slow.
It is also interesting to plot these intermediate values.

Full list of files:
- `_parameters.csv`: The design values of the example. There will be one row for each trap. The columns are `amplitude`, `tau_high`, and `tau_low` (there may be more columns for some aRTN cases).
- `_signals.feather`: The raw time-series signal with 1 million points. This file includes the "design value" signal for each trap (labelled `trap_{i}`), the sum of all pure RTN components (`rtn_sum`), and the white noise (`white_noise`). The algorithm is not allowed to look at these. They are only for calculating the accuracy after the algorithm has run. The main column is `full_signal`. This is the sum of `rtn_sum` and `white_noise`. There may also be some temporary columns added by certain steps.
- `_kde_data.feather`: The probability density function and other data output by the KDE step.

	I recommend using `df.squeeze()` after reading this file
	to convert the Pandas `DataFrame` into a `Series`.
	
	This file includes many columns:
	- `intensity`: Intensity is the **independent variable** of the KDE. This column will be an array of equispaced points along the axis of the intensity of the signal. It is like current, but we are trying to be agnostic to the measurement type (current/voltage/other). This has negative values because the intensity of the signal can sometimes be negative.
	- `density`: This is the probability density. It is the **dependent variable** for the KDE. This column is an array with each value being the probability of the intensity with the same index. This is a probability and should always be positive. The sum of this array should be 1.
	- `raw_density` and `raw_intensity`: These are the same as `density` and `intensity`, but for the unfiltered raw signal. These are not used by the algorithm, only to make the gray plot in the KDE figure.
	- `peaks_intensities`: This is a list of the intensities of the detected peaks. This is the seed data for the GMM.1
	- `peaks_densities`: This is a list of the densities of the detected peaks. This is not used by the algorithm. It is only saved to draw a marker over the peak in the KDE plot.
	- `window`: The width of the rolling mean used, which is a function of the estimated white noise.

- `_timer_kde.txt`: Time taken for the KDE step.
- `_decomp_data_traps.csv`: Result of the GMM step: the predicted amplitude of each trap. This is a dataframe with two columns: `trap` and `sep`. `sep` is the separation between two fitted Gaussian functions (i.e. the trap switching amplitude). `trap` is the trap index (arbitrary and may be different from the indices used elsewhere).
- `_sep_error.feather`: Accuracy of the GMM step: the error to predict the amplitude of each trap (**reads design values, cannot be used later in the algorithm**).
- `_gmm_fit.lmfit`: The full fitted [`lmfit`](https://lmfit.github.io/lmfit-py/) (the library used for the GMM) Gaussian mixture model.
- `_timer_decompose.txt`: Time taken for the GMM step.
- `_time_series_predictions.feather`: The raw predictions of the RNN step.
- `_tf_model.h5`: Weights of the trained RNN model.

The class `Example` in [`example.py`](./example.py) was created to help manage all of these files.
Since many of these files for a given example must be read,
this class allows you to quickly get all the different variants
just by accessing that attribute on the class.
It will also allow you to read and write the file
intelligently with different methods depending on the data type (`.csv`, `.feather`, `.h5`, `.txt` all have different functions to read and write).
Please see [`example.py`](./example.py) for the full capabilities.
For example:
```python
example = Example('<some_example>_signals.feather')
print(example.path)  # <some_example>_signals.feather
print(example.parameters.path)  # <some_example>_parameters.csv
print(example.parameters.read())  # DataFrame of the contents of <some_example>_parameters.csv
example.kde_data.write(pd.DataFrame())  # Saves an empty dataframe to <some_example>_kde_data.csv
```

Many files (the large ones) are saved in `.feather` files instead of `.csv`.
This is because `.csv` is very inefficient.
It is a text-based format, so when saving the file, every number has to be converted into text (which is time-inefficient)
and this text is saved on the disk (which is very space-inefficient).
When the file is read, the text must be parsed back into numbers, which is time-inefficient.
Therefore, we use `.feather` files, which are efficient binary files, solving both of these issues.
Some small files are still saved as `.csv` (short list of parameters, accuracy results, etc.) for convenience.

## Code structure

### The main algorithm (3 steps)

The main code to do the three steps of our algorithm is located in [`./main_algorithm/`](./main_algorithm/).
There are three subfolders for our three algorithm steps:

1. [`a_kde`](./main_algorithm/a_kde)
1. [`b_gmm`](./main_algorithm/b_gmm)
1. [`c_rnn`](./main_algorithm/c_rnn)
1. [`d_hmm`](./main_algorithm/d_hmm): Alternative digitization step using HMM

Here, the prefixes `a_`, `b_`, `c_` are used to order the steps because Python files may not start with a number.

Running the algorithm is currently done one step at a time.
Future steps use data saved in files by previous steps.
This has been very helpful while working on a single step,
but it would be nice to eventually add a method to run the algorithm start-to-finish.

#### Running the main algorithm

1. KDE: [`python main.py process-kde FILES`](./USAGE.md#process-kde)
1. GMM: [`python main.py process-gmm FILES`](./USAGE.md#process-gmm)
1. RNN: [`python main.py train FILES`](./USAGE.md#train)

Plase see [`USAGE.md`](./USAGE.md) for more information.

#### Viewing the results of the main algorithm

1. KDE: [`python main.py plot-kde FILES`](./USAGE.md#plot-kde)
1. GMM: Already handled by `process-gmm`
1. RNN: [`python main.py plot-time-series-predictions`](./USAGE.md#plot-time-series-predictions)

Plase see [`USAGE.md`](./USAGE.md) for more information.

#### Calculating accuracy for the run

1. Switching amplitude: [`python main.py aggregate-amplitude-error FILES`](./USAGE.md#aggregate-amplitude-error)
1. Digitization: [`python main.py aggregate-digitization-tau-error FILES`](./USAGE.md#aggregate-digitization-tau-error)

Plase see [`USAGE.md`](./USAGE.md) for more information.

### Data generation

The code to generate RTN data is in [`data_generation`](./data_generation).
This folder contains the files to generate validation data (like our 330 normal RTN dataset)
as well as files to generate training data used by the RNN (the two share the same core functions).

### Presentation of the results

The code to make "fancy" figures like those included in manuscripts
is in [`results_presentation`](./results_presentation).
There are more details in [the `README.md` of that folder](./results_presentation/README.md).

### Data exploration

The [`data_exploration`](./data_exploration) folder holds the code to generate
less fancy figures for data exploration.
Normally, these figures are also to understand the input data, not to present the results.

### Aggregation of the results

The folder [`results_aggregation`](./results_aggregation) holds files to aggregate the results from the intermediate files for each example
into a single file. These files are very messy due to the multiple different results formats.

### Experiments and debugging

The folder [`experiments_debugging`](./experiments_debugging) holds
temporary code, works in progress, and debugging code.

## Usage (running the code)

Please see [`USAGE.md`](./USAGE.md) for the full usage.
