
"""

TEST RANDOM 1

Statement:
A tumor type (cell type) produces more false negatives compared to
true positives than others.

Solution:
Compute the z-score of the tumor type and its corresponding p-value
at a significance level of 0.05 .
"""

from stat_tests import Result, significance_test

tumor_count = 10
trials = 10

# Generate some data
import random
tumors = []
for i in range(0, tumor_count):
    tumors.append({})
    tumors[i]["name"] = str(i)
    tumors[i][Result.TP] = random.randint(0, trials)
    tumors[i][Result.FP] = trials - tumors[i][Result.TP]

significance_test(tumors)
