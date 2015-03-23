import os

import constants as const
from utils import generate_summary_predicates


generate_summary_predicates(const.FOLDER_SORT_CORRECTNESS)
generate_summary_predicates(const.FOLDER_SORT_STABILITY)
