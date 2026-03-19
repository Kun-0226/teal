import torch
import torch.nn as nn
import torch_sparse
from .utils import weight_initialization


class FlowGNN(nn.Module):
    def __init__(self, teal_env, num_layer):
        super(FlowGNN, self).__init__()

        self.env = teal_env
        self.num_layer = num_layer

        # 基础拓扑参数
        self.edge_index = self.env.edge_index
        self.edge_index_values = self.env.edge_index_values
        self.num_path = self.env.num_path
        self.num_path_node = self.env.num_path_node
        self.num_edge_node = self.env.num_edge_node

        # 总节点数 N = 链路数 + 路径数
        self.N = self.num_edge_node + self.num_path_node

        # 预计算转置索引，用于 Path -> Edge 的反向消息传递
        # 必须维持 N x N 的全图索引范围
        self.edge_index_t = self.edge_index[[1, 0]]

        # 初始维度 dim 为 1
        self.initial_dim = 1

        self.edge_grus = nn.ModuleList()
        self.path_grus = nn.ModuleList()
        self.dnn_list = nn.ModuleList()

        for i in range(self.num_layer):
            curr_dim = i + 1
            self.edge_grus.append(nn.GRUCell(curr_dim, curr_dim))
            self.path_grus.append(nn.GRUCell(curr_dim, curr_dim))
            self.dnn_list.append(
                nn.Linear(self.num_path * curr_dim, self.num_path * curr_dim)
            )

        self.apply(weight_initialization)

    def forward(self, h_0):
        """
        h_0 形状: [N, 1]
        """
        h_i = h_0

        for i in range(self.num_layer):
            # --- 步骤 1: 全图消息聚合 (Edge -> Path) ---
            # 使用原始全图索引，避免切片导致的越界
            # 聚合结果 msg 形状为 [N, curr_dim]
            msg = torch_sparse.spmm(
                self.edge_index,
                self.edge_index_values,
                self.N,
                self.N,
                h_i
            )

            # --- 步骤 2: 仅更新 Path 节点的隐藏状态 ---
            # Path 节点位于矩阵的后半部分
            h_path_old = h_i[-self.num_path_node:, :]
            msg_path = msg[-self.num_path_node:, :]
            h_path_new = self.path_grus[i](msg_path, h_path_old)

            # --- 步骤 3: 全图反向聚合 (Path -> Edge) ---
            # 使用转置索引将路径信息带回给链路
            msg_t = torch_sparse.spmm(
                self.edge_index_t,
                self.edge_index_values,
                self.N,
                self.N,
                h_i  # 这里可以用更新前的，也可以用临时组合的
            )

            # --- 步骤 4: 仅更新 Edge 节点的隐藏状态 ---
            # Edge 节点位于矩阵的前半部分
            h_edge_old = h_i[:-self.num_path_node, :]
            msg_edge = msg_t[:-self.num_path_node, :]
            h_edge_new = self.edge_grus[i](msg_edge, h_edge_old)

            # --- 步骤 5: Demand 约束处理 (Path 节点线性层) ---
            curr_dim = h_path_new.shape[1]
            h_path_node_dnn = self.dnn_list[i](
                h_path_new.reshape(
                    self.num_path_node // self.num_path,
                    self.num_path * curr_dim
                )
            ).reshape(self.num_path_node, curr_dim)

            # 重新组合全图特征 [Edge_new + Path_new]
            h_i = torch.cat([h_edge_new, h_path_node_dnn], dim=0)

            # --- 步骤 6: Skip Connection (维度增长逻辑与原版一致) ---
            h_i = torch.cat([h_i, h_0], dim=-1)

        # 最终输出最后 num_path_node 行，维度 [num_path_node, num_layer + 1]
        return h_i[-self.num_path_node:, :]