
"""

TEST FILE 1

Statement:
A tumor type (cell type) produces more false negatives compared to
true positives than others.

Solution:
Compute the z-score of the tumor type and its corresponding p-value
at a significance level of 0.05 .
"""

from pprint import pprint

from stat_tests import Result, significance_test

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description=
                                     "Run a significance test.")
    parser.add_argument("data",
                        help="The data TSV file.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Be verbose.")
    args = parser.parse_args()
    argv = vars(args)
    return argv

def get_XPs(filename):
    """
    Read/return the TPs and FPs
    If TPs == FPs == 0 , we ignore the record
    """
    ignored = 0
    results = []
    with open(filename) as fp:
        line = fp.readline() # Discard header
        while True:
            line = fp.readline()
            if len(line) == 0: break
            p1 = line.find("\t")
            if p1 < 0: raise Exception("Bad data!")
            p2 = line.find("\t", p1+1)
            if p2 < 0: raise Exception("Bad data!")
            p3 = line.find("\t", p2+1)
            if p3 < 0: raise Exception("Bad data!")
            p4 = line.find("\t", p3+1)
            if p4 < 0: raise Exception("Bad data!")
            name = line[0:p1]
            tps_s = line[p1+1:p2]
            tps = int(float(tps_s))
            fps_s = line[p3+1:p4]
            fps = int(float(fps_s))
            # print("%-20s %3i %3i" % (name, tps, fps))
            if tps == 0 and fps == 0:
                ignored += 1
                continue
            results.append({ "name" : name,
                             Result.TP : tps,
                             Result.FP : fps })
    print("ignored: %i" % ignored)
    return results


argv = parse_args()
tumors = get_XPs(argv["data"])
# pprint(tumors)
significance_test(tumors)
