"""
Configuration parameters.
"""

import os

DATA_PATH=os.environ.get('DATA_PATH', "/data")
RESOURCES_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')