
import argparse
import logging
import sys


class UserError(Exception):
    pass


def main():
    try:
        args = parse_args()
        logger = setup_logging(args)
        validate_args(args, logger)

        header, input_lines = read_input(args, logger)
        logger.info("original data size: %i" % len(input_lines))

        table = read_table(args)
        logger.info("table size: %i" % len(table))

        index = make_index(args, logger, table)

        run_query(args, logger, header, input_lines, table, index)

    except UserError as e:
        logger.fatal("cell_grep: user error: " + " ".join(e.args))
        exit(1)


def parse_args():
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
    parser.add_argument("output", nargs="?",
                        help="Output CSV, required unless doing a count")
    parser.add_argument("-n", "--negate", action="store_true",
                        help="Negate the match")
    parser.add_argument("-c", "--count", action="store_true",
                        help="Simply report counts, do not create CSV")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Decrease verbosity")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Increase verbosity")
    args = parser.parse_args()
    return args


def setup_logging(args):
    logger = logging.getLogger("cell_grep")
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARN)
    else:
        logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter("%(levelname)-5s %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def validate_args(args, logger):
    if logger.isEnabledFor(logging.DEBUG):
        V = vars(args)
        for a in V:
            logger.debug("%-8s %s" % (a + ":", V[a]))

    if not args.count and args.output is None:
        raise UserError("If not counting, " +
                        "you must provide an output file")


def read_input(args, logger):
    """
    Return header: first line of input, containing CSV column headers
           lines:  list of str for rest of file
    """
    with open(args.data, "r") as fp:
        # First line is a header
        header = fp.readline()
        # Check format:
        if "DATASET" not in header:
            raise UserError("'DATASET' not in header")
        tokens = header.split(",")
        # Name must be in column 4!
        if tokens[4] != "CELL_LINE_NAME":
            raise UserError("'CELL_LINE_NAME' must be in column 4")
        # Read the rest of the file normally:
        lines = fp.readlines()
    return header, lines


def read_table(args):
    """ Return list of str for file lines """
    with open(args.table, "r") as fp:
        table = fp.readlines()
    return table


def make_index(args, logger, table):
    """ Make a dict index that maps str alias to int line number """
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
    return index


def run_query(args, logger, header, input_lines, table, index):
    """ Do the real work """
    found_count = 0
    total_count = 0
    select_count = 0

    if args.output is None:
        fp = sys.stdout
    else:
        fp = open(args.output,  "w")

    if not args.count:
        fp.write(header)

    for line in input_lines:
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
        matched = (args.pattern in row) ^ args.negate
        if matched:
            select_count += 1
            if not args.count:
                fp.write(line)

    msg = ""
    if args.count or logger.isEnabledFor(logging.INFO):
        msg = "found: %i , not found: %i , selected: %i" % \
              (found_count, total_count-found_count, select_count)

    if args.count:
        print(msg)
    else:
        logger.info(msg)


if __name__ == "__main__":
    main()
