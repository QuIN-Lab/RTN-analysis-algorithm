
# Usage
```
Usage: python -m main [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  generate-data                     Generate RTN data of different varieties.
  hmm                               Predict the underlying RTN component using HMM for...
  output-time-series                Output time series data using pre-trained models for...
  process-gmm                       Run GMM on all examples specified by FILES (specify...
  process-kde                       Run KDE step for the examples specified by FILES...
  tau-fitting                       Plot the tau histogram and the e^(-x/tau) / tau...
  train                             Train the RNN model for the examples specified by...
```

## `generate-data`

Generate RTN data of different varieties.


```
Usage: python -m main generate-data [OPTIONS]

Options:
  --examples INTEGER              Number of examples to generate.  [required]
  --signal-length INTEGER         Length of signal to generate  [required]
  --noise TEXT                    Level of white noise to add to the signal. Can either be the absolute value,
                                  or the low and high value (separated by `-`) to generate randomly.
                                  [required]
  --variety [metastable|missing-level|coupled|normal]
                                  [required]
  --traps INTEGER                 The number of traps  [required]
  --out-dir PATH                  Where to save the data  [required]
  --help                          Show this message and exit.

```

## `hmm`

Predict the underlying RTN component using HMM
for all signals specified by FILES (provide `_signals.feather` files).
Write the time-series predictions to RESULTS_DIR.
This uses HMM, not fHMM, so it only really works for 1-trap data.


```
Usage: python -m main hmm [OPTIONS] FILES... RESULTS_DIR

Options:
  --help  Show this message and exit.

```


## `output-time-series`

Output time series data using pre-trained models
for the examples specified by FILES (specify `_signals.feather` files).
Save the results in RESULTS_DIR.


```
Usage: python -m main output-time-series [OPTIONS] FILES... RESULTS_DIR

Options:
  -m, --mode [normal|metastable|coupled|missing-level|mutually-exclusive|auto]
                                  Algorithm mode. If `auto`, detect automatically from filename.  [required]
  -n, --n-traps [1|2|3|auto]      The number of traps. If `auto`, determine from filename.  [required]
  -l, --recurrent-layer [gru|lstm]
                                  Type of recurrent layer to use (RNN only)  [required]
  --model-type [rnn|wavenet]      Model structure to use  [required]
  --help                          Show this message and exit.

```


## `process-gmm`

Run GMM on all examples specified by FILES (specify `_signals.feather`
files).


```
Usage: python -m main process-gmm [OPTIONS] FILES...

Options:
  -m, --mode [normal|metastable|coupled|missing-level|mutually-exclusive|auto]
                                  Algorithm mode. If `auto`, detect automatically from filename.  [required]
  --output-format [pdf|png]
  --debug / --no-debug
  --help                          Show this message and exit.

```

### Example 1: 
To process the GMM and generate the GMM plot, run the command:

`python -m main process-gmm "~/OneDrive/02. QuIN_Research/31. Noise-RTN/02. 2022_2021_Algorithm paper/simulated_rtn/2021_07_23_generated_normal_rtn_data_4_white_noise_study/1-trap_wn=0.4_example=0_signals.feather" --output-format=png`
![](docs/command-output-example-images/process-gmm_1-trap_wn=0.4_example=0_decomposition.png)

## `process-kde`

Run KDE step for the examples specified by FILES
(specify `_signals.feather` or `_signals.csv` files).


```
Usage: python -m main process-kde [OPTIONS] FILES...

Options:
  --mode [normal|metastable|coupled|missing-level|mutually-exclusive|auto]
                                  The algorithm mode. If `auto`, determine from filename.  [required]
  --input-column TEXT             The column of the input dataframe to consider as the raw, noisy input
                                  signal.
  --help                          Show this message and exit.

```

### Example 1: Process single file
To run the KDE step on a file, execute this command (the path depends on where you saved the data on your computer).

`python -m main process-kde "~/OneDrive/02. QuIN_Research/31. Noise-RTN/01. 2021_Algorithm paper/simulated_rtn/2021_07_23_generated_normal_rtn_data_4_white_noise_study/1-trap_wn=0.0_example=0_signals.feather"`


### Example 2: Process multiple files

You can also process multiple examples in parallel by specifying a pattern
rather than a single file. The example below uses the wildcard character `*`
to match any file in the white noise study folder ending in
`_signals.feather`.


`python -m main process-kde "~/OneDrive/02. QuIN_Research/31. Noise-RTN/01. 2021_Algorithm paper/simulated_rtn/2021_07_23_generated_normal_rtn_data_4_white_noise_study/*_signals.feather"`



## `train`

Train the RNN model for the examples specified by FILES (specify
`_signals.feather` files).
Save the results (trained model, full predictions, accuracies) in
RESULTS_DIR.

For simulated signals, the default options should be sufficient.
All required information will be extracted from the filename.

For real measurements, please specify `--mode` based on
your assessment of the aRTN case.

This command has replaced the previous `train-artn` and `train-ctn` commands
(cc7af0e1).


```
Usage: python -m main train [OPTIONS] FILES... RESULTS_DIR

Options:
  -m, --mode [normal|metastable|coupled|missing-level|mutually-exclusive|auto]
                                  Algorithm mode. If `auto`, detect automatically from filename.  [required]
  --retrain / --no-retrain        Wether to retrain or to load weights from the saved .h5 file  [required]
  -n, --n-traps [1|2|3|auto]      The number of traps. If `auto`, determine from filename.Only required for
                                  simulated data to calculate accuracy.  [required]
  --skip-existing / --no-skip-existing
                                  Skip training if the time_series_predictions file exists.
  -l, --recurrent-layer [gru|lstm]
                                  Type of recurrent layer to use (RNN only)  [required]
  --model-type [rnn|wavenet]      Model structure to use  [required]
  --help                          Show this message and exit.

```

### Example 1: 
The RNN is run in a similar way, although there are many more options (like GRU vs. LSTM).

`python -m main train "~/OneDrive/02. QuIN_Research/31. Noise-RTN/02. 2022_2021_Algorithm paper/simulated_rtn/2021_07_23_generated_normal_rtn_data_4_white_noise_study/1-trap_wn=0.4_example=0_signals.feather"`


### Example 2: 
The mode must be guessed when running on real measurements.

`python -m main train --mode=missing-level "~/OneDrive/02. QuIN_Research/31. Noise-RTN/02. 2022_2021_Algorithm paper/simulated_rtn/30. CNT real data/01.LBm1_Set1_1um_500nm_3.7K_Sampling_2.0V_6.1uA_9sets_Run101-Run109_signals.feather"`

