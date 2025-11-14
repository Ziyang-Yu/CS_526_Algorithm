import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from time import time
from data.generator import gen_graph
from core.sssp_main import sssp_break_sorting
from baseline.dijkstra import dijkstra

g = gen_graph(2000)
t0=time(); x=sssp_break_sorting(g,2000,0); print("Ours:",time()-t0)
t0=time(); y=dijkstra(g,0); print("Dijkstra:",time()-t0)
