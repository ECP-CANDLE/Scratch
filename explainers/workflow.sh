#!/bin/bash
set -eu

source env.sh

swift-t workflow.swift $*
