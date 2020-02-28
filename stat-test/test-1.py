
"""
Statement:
A tumor type (cell type) produces more false negatives compared to
true positives than others.

Solution:
Compute the z-score of the tumor type and its corresponding p-value
at a significance level of 0.05 .
"""

from enum import Enum
class Result(Enum):
    TP = 1
    TN = 2
    FP = 3
    FN = 4

    def __str__(self):
        return self.name
    def __repr__(self):
        return self.name


tumors = 10
trials = 10

# Generate some data
import random
tumor = []
for i in range(0, tumors):
    tumor.append({})
    tumor[i][Result.TP] = random.randint(0, trials)
    tumor[i][Result.FN] = trials - tumor[i][Result.TP]

# Construct the Numpy data
import numpy as np
data = np.zeros(tumors, dtype=np.float64)
for i in range(0, tumors):
    data[i] = tumor[i][Result.FN] / tumors

# Report basic statistics
# print(data)
print("mean: %0.3f" % data.mean())
print("std:  %0.3f" % data.std())

def print3(data):
    """ Print 3 significant digits """
    import sys
    sys.stdout.write("[")
    values = [ f"{v: 0.3f}" for v in data ]
    sys.stdout.write(", ".join(values))
    sys.stdout.write("]\n")

# Compute the p-values
import scipy.stats
zscores = scipy.stats.zscore(data)
# print3(zscores)
p_values = scipy.stats.norm.cdf(zscores)
# print3(p_values)

# Report the lowest p-value
best = np.argmin(p_values)
print((f"lowest p-value sample: "
       f"index={best} "
       f"value={data[best]:0.3f} "
       f"z-score={zscores[best]:0.3f} "
       f"p-value={p_values[best]:0.3f}"))

if p_values[best] < 0.05:
    print("sample is statistically significant")
else:
    print("sample is not statistically significant")
