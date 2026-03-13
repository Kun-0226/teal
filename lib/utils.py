import sys
from contextlib import contextmanager

import torch
import torch.nn as nn


def weight_initialization(module):
    """Initialize weights in nn module"""

    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight, gain=1)
        torch.nn.init.constant_(module.bias, 0)


def uni_rand(low=-1, high=1):
    """Uniform random variable [low, high)"""
    return (high - low) * np.random.rand() + low


def print_(*args, file=None):
    """print out *args to file"""
    if file is None:
        file = sys.stdout
    print(*args, file=file)
    file.flush()


import json
import networkx as nx
from networkx.readwrite import json_graph


def compute_local_backup_paths(topo_json_data, max_paths=4):
    """
    计算任意相邻节点对（即每一条链路）的局部备份路径
    :param topo_json_data: 包含拓扑信息的 dict
    :param max_paths: 每条链路最多保留的备份路径数量 (默认 4)
    :return: dict，格式为 {(u, v): [[(u, x), (x, y), (y, v)], ...]}
    """
    # 1. 从 JSON 数据读取有向图
    G = json_graph.node_link_graph(topo_json_data)

    local_backup_paths = {}

    # 2. 遍历每一条相邻节点（即每一条物理链路）
    for u, v in list(G.edges()):
        # 备份该边的属性，然后暂时从图中移除这条“故障边”
        edge_data = G.edges[u, v].copy()
        G.remove_edge(u, v)

        backup_paths_for_uv = []
        try:
            # 3. 在残余图中寻找从 u 到 v 的“边不相交”备份路径
            # nx.edge_disjoint_paths 会返回多条独立的路径
            disjoint_paths_generator = nx.edge_disjoint_paths(G, u, v)

            # 取出最多 max_paths 条路径
            for i, node_path in enumerate(disjoint_paths_generator):
                if i >= max_paths:
                    break

                # 将节点序列 [u, node1, node2, v] 转换为边序列 [(u, node1), (node1, node2), (node2, v)]
                edge_path = [(node_path[k], node_path[k + 1]) for k in range(len(node_path) - 1)]
                backup_paths_for_uv.append(edge_path)

        except nx.NetworkXNoPath:
            # 如果剔除该边后，u 和 v 之间没有连通路径了，就保留空列表
            pass

        # 4. 把算好的备份路径存入字典
        local_backup_paths[(u, v)] = backup_paths_for_uv

        # 5. 将该边恢复回图中，准备计算下一条边
        G.add_edge(u, v, **edge_data)

    return local_backup_paths
