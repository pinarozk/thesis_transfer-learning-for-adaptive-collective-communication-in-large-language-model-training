import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CCL = BASE_DIR / "external" / "ilp_ccl_tree"

sys.path.append(str(CCL))

from tree_generate import *
from topo_generate import *
from network_parameter import *
from results_process import *

print("OK")