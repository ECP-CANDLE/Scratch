
# STAT TESTS PY

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


def significance_test(tumors):
    """
    tumors: an array of dicts:
             each dict has keys TP, FP
             with corresponding integer counts
    """

    import numpy as np

    tumor_count = len(tumors)

    # Construct the Numpy data
    data = np.zeros(tumor_count, dtype=np.float64)
    for i in range(0, tumor_count):
        trials = tumors[i][Result.FP] + tumors[i][Result.TP]
        if trials != 0:
            data[i] = tumors[i][Result.FP] / trials
        else:
            data[i] = np.nan

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
           f"name='{tumors[best]['name']}' "
           f"value={data[best]:0.3f} "
           f"z-score={zscores[best]:0.3f} "
           f"p-value={p_values[best]:0.3f}"))

    # We use HIT/MISS for easier grepping later
    if p_values[best] < 0.05:
        print("HIT:  sample is statistically significant")
    else:
        print("MISS: sample is not statistically significant")
