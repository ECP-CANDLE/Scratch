#!/bin/zsh
set -eu

# GET DATA ZSH
# Get the TSVs from the NCI Globus endpoint

# petrel#ncipilot
NCIPILOT=ebf55996-33bf-11e9-9fa4-0a06afd4a22e

# A local machine to receive the transfer
DUNEDIN=5eeddf3e-573e-11ea-971b-021304b0cca7

# Endpoint data directory
PREDICTIONS=public/Sample_Evaluation_for_Classification/Predictions

# Local data directory
DESTINATION=usb2/wozniak/CANDLE-Bulk/stat-test/Predictions

# List of all top_ directories:
TOPS=$( globus ls $ncipilot:$predictions  --filter "~top21_*" )
TOPS=( ${=TOPS} )

print "TOPS: ${#TOPS}"

TASKS=()

mkdir -pv /${DESTINATION}
cd /${DESTINATION}
# For each top_ directory:
for TOP in ${TOPS}
do
  # Make the local directory:
  mkdir -pv ${TOP}
  cd ${TOP}
  # List the TSVs:
  TSVS=$( globus ls ${NCIPILOT}:${PREDICTIONS}/${TOP} --filter "~*.tsv" )
  TSVS=( ${=TSVS} )
  print ${NCIPILOT}:${PREDICTIONS}/${TOP}
  print -l ${TSVS} > list.txt
  # We don't want the big all.tsv:
  sed -i 's/all/#all/' list.txt
  # Format for Globus:
  paste list.txt list.txt > pairs.list
  # Transfer them! (in the background)
  globus transfer \
         ${NCIPILOT}:${PREDICTIONS}/${TOP} \
         ${DUNEDIN}:${DESTINATION}/${TOP} \
         --batch < pairs.list > globus.txt
  # Extract the Task ID
  TASK=$( sed -n '/ID:/{s/Task ID: \(.*\)/\1/;p}' globus.txt )
  TASKS+=${TASK}
  cd -
done

print TASKS: ${TASKS}

# Wait for all transfers
for TASK in ${TASKS}
do
  print "WAIT: ${TASK}"
  globus task wait ${TASK}
done

print OK
