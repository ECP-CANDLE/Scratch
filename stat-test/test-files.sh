#!/bin/bash
set -eu

# TEST FILES SH
# Perform the stats test on all supported TSVs
#         in given directory DATA

# Get command-line argument:
if [[ ${#} != 1 ]]
then
  echo "Provide the DATA directory!"
  exit 1
fi
DATA=$1

# Find the TSVs
TSVS=$( find -H $DATA \
             -name 'ccounts.tsv'   -o \
             -name 'moacounts.tsv' -o \
             -name 'ctypecounts.tsv' )
TSVS=( $TSVS )

# Scan the TSVs
echo "TSVS: ${#TSVS}"
for TSV in ${TSVS[@]}
do
  echo "TSV: $TSV"
  DIR=$(   dirname  $TSV )
  LABEL=$( basename --suffix=.tsv $TSV )
  python test-file-1.py $TSV > $DIR/$LABEL.stats
done
