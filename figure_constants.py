"""
Constants used for generating figures, to keep consistent font sizes and things
between all the figures in the paper.

Developed by Marcel Robitaille on 2022/02/18 Copyright © 2021 QuIN Lab
"""

import matplotlib.pyplot as plt

plt.switch_backend('Agg')
plt.style.use(['science'])

# EDL submission (very space limited)
# LABEL_FONT_SIZE = 9

# APS poster has more space
LABEL_FONT_SIZE = 10

TICK_PADDING = 2

RAW = '#999'  # gray
FILTERED = '#0c5da5'  # C0
GMM = '#00b945'
DIGITIZED = 'orange'
