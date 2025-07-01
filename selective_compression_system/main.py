"""
Comprehensive Benchmarking Suite for Selective Memory Compression System
========================================================================
compression system using real GPT-4o-mini API calls and synthetic data generation.
"""


import os
import sys
import json
import csv
import time
import logging
import random
import statistics
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import defaultdict, Counter

# Import the memory compression system
sys.path.append('selective_compression_system')
from core.system import SelectiveCompressionSystem
from core.memory import MemorySegment

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
