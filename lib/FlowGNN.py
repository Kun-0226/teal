import math

import torch
import torch.nn as nn
import torch_scatter
import torch_sparse

from .utils import weight_initialization


class FlowGNN(nn.Module):
    """Transform the demands into compact feature vectors known as embeddings.

    FlowGNN alternates between
    - GNN layers aimed at capturing capacity constraints;
    - DNN layers aimed at capturing demand constraints.

    Replace torch_sparse package with torch_geometric pakage is possible
    but require larger memory space.
    """

    def __init__(self, teal_env, num_layer):
        """Initialize flowGNN with the network topology.

        Args:
            teal_env: teal environment
            num_layer: num of layers in flowGNN
        """

        super(FlowGNN, self).__init__()

        self.env = teal_env
        self.num_layer = num_layer

        self.edge_index = self.env.edge_index# 稀疏邻接矩阵（PathNode ↔ EdgeNode 的二部图）
        self.edge_index_values = self.env.edge_index_values# edge_index_values：权重（是否在路径上）
        self.num_path = self.env.num_path # 每个 demand 的路径数
        self.num_path_node = self.env.num_path_node
        self.num_edge_node = self.env.num_edge_node
        # self.adj_adj = torch.sparse_coo_tensor(self.edge_index,
        #    self.edge_index_values,
        #    [self.num_path_node + self.num_edge_node,
        #    self.num_path_node + self.num_edge_node])

        self.gnn_list = []
        self.dnn_list = []
        for i in range(self.num_layer):
            # to replace with GCNConv package:
            # self.gnn_list.append(GCNConv(i+1, i+1))
            self.gnn_list.append(nn.Linear(i+1, i+1))
            self.dnn_list.append(
                nn.Linear(self.num_path*(i+1), self.num_path*(i+1)))
        self.gnn_list = nn.ModuleList(self.gnn_list)
        self.dnn_list = nn.ModuleList(self.dnn_list)

        # weight initialization for dnn and gnn
        self.apply(weight_initialization)

    def forward(self, h_0):
        """Return embeddings after forward propagation

        Args:
            h_0: inital embeddings
        """

        h_i = h_0
        for i in range(self.num_layer):
            """
            参数	含义 & 细节
self.edge_index	图的边索引，是 [2, E] 形状的张量（COO 格式），E 是边数；第一行是源节点（u），第二行是目标节点（v），表示边 u → v；例：[[0,1,2],[1,2,0]] 表示边 0→1、1→2、2→0。
self.edge_index_values	稀疏矩阵 A 的非零元素值（边权重），[E] 形状的张量；如果是无权重图，通常全为 1；如果是有权重图，对应每条边的权重。
h_0.shape[0]	稀疏矩阵 A 的行数 = 图的节点总数 N（h_0 是初始节点特征，shape[0] 是节点数）。
h_0.shape[0]	稀疏矩阵 A 的列数 = 图的节点总数 N（邻接矩阵是 N×N 的方阵）。
h_i	待聚合的节点特征矩阵，形状为 [N, D]（N 节点数，D 特征维度）；是乘法中的稠密矩阵 X，输出形状仍为 [N, D]。
            """

            # gnn
            # to replace with GCNConv package:
            # h_i = self.gnn_list[i](h_i, self.edge_index)

            h_i = self.gnn_list[i](h_i)#线性层
            # h_i = torch.sparse.mm(self.adj_adj, h_i)
            #过稀疏矩阵乘法，利用图的边信息（edge_index/edge_index_values），对节点特征 h_i 进行聚合（邻居信息传递），最终得到更新后的节点特征 h_i
            h_i = torch_sparse.spmm(
                self.edge_index,  # 参数1：稀疏矩阵A的COO格式索引
                self.edge_index_values,  # 参数2：稀疏矩阵A的非零元素值（边权重）
                h_0.shape[0],  # 参数3：稀疏矩阵A的行数（节点总数N）
                h_0.shape[0],  # 参数4：稀疏矩阵A的列数（节点总数N）
                h_i  # 参数5：稠密矩阵X（待聚合的节点特征）
            )
            # dnn
            #取最后num_path_node行，即只看path embedding的结果
            #再reshape成[demand数 , path特征拼接]
            #经过线性层后在reshape回[num_path_node , feature]
            h_i_path_node = self.dnn_list[i](
                h_i[-self.num_path_node:, :].reshape(
                    self.num_path_node//self.num_path,
                    self.num_path*(i+1)))\
                .reshape(self.num_path_node, i+1)
            #拼接恢复完整节点[EdgeNode , PathNode]
            h_i = torch.concat(
                [h_i[:-self.num_path_node, :], h_i_path_node], axis=0)

            # skip connection
            #保留原始信息
            h_i = torch.cat([h_i, h_0], axis=-1)
        # 最终只返回PathNode embedding
        # return path-node embeddings
        return h_i[-self.num_path_node:, :]
