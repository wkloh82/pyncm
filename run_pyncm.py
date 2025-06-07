#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyNCM Wrapper Script
This script makes it easier to run PyNCM by handling the Python path setup.
"""

import os
import sys

# Add the parent directory of pyncm to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import and run the main function
try:
    from pyncm.__main__ import __main__
    __main__()
except ImportError as e:
    print(f"Error: {e}")
    print("\nPython path:")
    for p in sys.path:
        print(f"  {p}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1) 