
import argparse

parser = argparse.ArgumentParser(
    prog="cell-grep",
    description=
    """
    Select keys from GDSC that match or don't.
    The pattern is for a simple substring match.
    Input: GDSC2 table.txt Output: selection.txt
    """)

parser.add_argument("data")
parser.add_argument("table")
parser.add_argument("pattern")
parser.add_argument("-n", "--negate", action="store_true", help="Negate the match")
parser.add_argument("output")

args = parser.parse_args()

with open(args.data, "r") as fp:
    # First line is a header
    header = fp.readline()
    # Check format:
    assert "DATASET" in header
    tokens = header.split(",")
    # Name must be in column 4!
    assert tokens[4] == "CELL_LINE_NAME"
    # Read the rest of the file normally:
    original = fp.readlines()
print("original data size: %i" % len(original))

with open(args.table, "r") as fp:
    table = fp.readlines()
print("table size: %i" % len(table))
# index maps string alias to line number
index = {}
i = 0
for entry in table:
    entry = entry.strip()
    tokens = entry.split("\t")
    index[tokens[0]] = i
    aliases = tokens[4].split(",")
    for alias in aliases:
        index[alias] = i
        # print("alias: '%s' line: %i" % (alias, i))
    i += 1

found_count = 0
total_count = 0
select_count = 0

fp = open(args.output,  "w")
fp.write(header)

for line in original:
    total_count += 1
    tokens = line.split(",")
    name = tokens[4]
    found = False
    if name in index:
        found = True
        row = table[index[name]]
    else:
        # print("not found: " + name)
        # exit()
        continue
    found_count += 1
    match = (args.pattern in row) ^ args.negate
    if match:
        select_count += 1
        fp.write(line)

print("found: %i , not found: %i , selected: %i" %
      (found_count, total_count-found_count, select_count))
