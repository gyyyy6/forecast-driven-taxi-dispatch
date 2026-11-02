"""入口脚本：构建 Taxi Zone 263×263 邻接矩阵。

推荐命令::

    python build_graph.py --config configs/default.yaml --method centroid
    python build_graph.py --method od --out data/processed/adj_matrix_od.npy
    python build_graph.py --method hybrid --alpha 0.5

默认输出：``data/processed/adj_matrix.npy``（行归一化 + 自环）。
"""
from __future__ import annotations

from src.features.graph import main

if __name__ == "__main__":
    main()