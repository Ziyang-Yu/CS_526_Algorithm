import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from time import time
from data.generator import gen_graph
from ablation.disable_pivots import sssp_no_pivots
from ablation.disable_partitioning import sssp_no_partition

g = gen_graph(2000)
t0=time(); sssp_no_pivots(g,2000,0); print("No pivots:",time()-t0)
t0=time(); sssp_no_partition(g,2000,0); print("No partition:",time()-t0)
