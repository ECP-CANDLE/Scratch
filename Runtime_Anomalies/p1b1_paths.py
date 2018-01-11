#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# =============================================================================
# Assumes Scratch is located at the same level as Benchmarks
# GitHub
#    Scratch
#        Runtime_Anomalies
#    Benchmarks
#        common
#        Pilot1
#            common
#            P1B1
# =============================================================================

import os
import sys

# =============================================================================
# Add paths to Benchmarks to system paths to allow imports
# =============================================================================
    
file_path = os.path.dirname(os.path.realpath(__file__))

paths = {"common" : ['..', '..', 'Benchmarks', 'common'],
         "P1_common" : ['..', '..', 'Benchmarks', 'Pilot1', 'common'],
         "P1B1" : ['..', '..', 'Benchmarks', 'Pilot1', 'P1B1']
        }

for path in paths.values():
    lib_path = os.path.abspath(os.path.join(*[file_path]+path))
    sys.path.append(lib_path)

