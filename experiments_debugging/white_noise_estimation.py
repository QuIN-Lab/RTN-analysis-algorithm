"""
This was the file use to experiment with different methods of estimating the
white noise present in the signal

Developed by Marcel Robitaille on 2021/11/03 Copyright © 2021 QuIN Lab
"""

import re
from glob import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


SAMPLE_WIDTH = 10
NUM_SAMPLES = 10

np.mean([
    df['full_signal'].loc[i:i+SAMPLE_WIDTH].std()
    for i in np.linspace(0, len(df) - SAMPLE_WIDTH - 1, NUM_SAMPLES)
])



def calculate_variances(f):
    df = pd.read_csv(f)
    # fig = px.line(df['full_signal'])
    # fig.show()
    # fig.write_image('fig1.svg')
    ratio = float(re.search(r'_wn=(\d+\.\d+)', f).group(1))
    theory = df.filter(regex='trap_.', axis=1).max().min() * ratio

    guess = np.mean([
        df['full_signal'].loc[i:i+SAMPLE_WIDTH].std()
        for i in np.linspace(0, len(df) - SAMPLE_WIDTH - 1, NUM_SAMPLES)
    ])

    return theory, df['white_noise'].std(), guess
    # print(df['white_noise'].var())


df = pd.DataFrame((
    calculate_variances(f)
    for f in glob('./data/*signals.csv')),
    columns=['theory', 'actual', 'guess'],
)
print(df)

# for f in glob('./data/2-*signals.csv'):
#     df = pd.read_csv(f)
#     # parameters = pd.read_csv(f.replace('signals', 'parameters'))
#     # print(parameters)
#     print()
#     break

# %%


# fix, ax = plt.subplots()
# plt.plot((df['theory'] - df['guess']) / df['theory'])
df.plot(alpha=0.7, figsize=(10, 7))
plt.tight_layout()
plt.xlim(0, None)
plt.ylim(0, None)
plt.savefig('./test.png', dpi=300)

# %%

df.to_csv('white_noise_estimation.csv', index=False)
