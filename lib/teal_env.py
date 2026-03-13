import pickle
import json
import os
import math
import time
import random
from itertools import product

from networkx.readwrite import json_graph

import torch
import torch_scatter
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

from .config import TOPOLOGIES_DIR
from .ADMM import ADMM
from .path_utils import find_paths, graph_copy_with_edge_weights, remove_cycles
from.utils import compute_local_backup_paths

class TealEnv(object):

    def __init__(
            self, obj, topo, problems,
            num_path, edge_disjoint, dist_metric, rho,
            train_size, val_size, test_size, num_failure, device,
            raw_action_min=-10.0, raw_action_max=10.0,failed_link=[]):
        """Initialize Teal environment.

        Args:
            obj: objective，优化目标
            topo: topology name
            problems: problem list
            num_path: number of paths per demand
            edge_disjoint: whether edge-disjoint paths
            dist_metric: distance metric for shortest paths 距离衡量指标
            rho: hyperparameter for the augumented Lagranian
            train size: train start index, stop index
            val size: val start index, stop index
            test size: test start index, stop index
            device: device id
            raw_action_min: min value when clamp raw action
            raw_action_max: max value when clamp raw action
        """

        self.obj = obj
        self.topo = topo
        self.problems = problems
        self.num_path = num_path
        self.edge_disjoint = edge_disjoint
        self.dist_metric = dist_metric

        self.train_start, self.train_stop = train_size
        self.val_start, self.val_stop = val_size
        self.test_start, self.test_stop = test_size
        self.num_failure = num_failure
        self.device = device

        # init matrices related to topology
        self.G = self._read_graph_json(topo)
        #self.capacity 是链路容量（EdgeNode）
        self.capacity = torch.FloatTensor(
            [float(c_e) for u, v, c_e in self.G.edges.data('capacity')])
        self.num_edge_node = len(self.G.edges)
        self.num_path_node = self.num_path * self.G.number_of_nodes()\
            * (self.G.number_of_nodes()-1)
        self.edge_index, self.edge_index_values, self.p2e = \
            self.get_topo_matrix(topo, num_path, edge_disjoint, dist_metric)

        # init ADMM
        self.ADMM = ADMM(
            self.p2e, self.num_path, self.num_path_node,
            self.num_edge_node, rho, self.device)

        # min/max value when clamp raw action
        self.raw_action_min = raw_action_min
        self.raw_action_max = raw_action_max

        self.reset('train')

        # new add
        self.edge2idx_dict # 链路到索引的映射，tuple(,)->int
        with open(os.path.join(TOPOLOGIES_DIR, topo)) as f:
            topo_data = json.load(f)
        self.local_backup_paths = compute_local_backup_paths(topo_data, max_paths=4)
        #self.local_backup_path={} #局部备份路径，dict{tuple(u,v):[[tuple(u,v),edge2...],path2...]}
        self.failed_link=failed_link

    def reset(self, mode='test'):
        """Reset the initial conditions in the beginning."""

        if mode == 'train':
            self.idx_start, self.idx_stop = self.train_start, self.train_stop
        elif mode == 'test':
            self.idx_start, self.idx_stop = self.test_start, self.test_stop
        else:
            self.idx_start, self.idx_stop = self.val_start, self.val_stop
        self.idx = self.idx_start
        self.obs = self._read_obs()

    def get_obs(self):
        """Return observation (capacity + traffic matrix)."""

        return self.obs
#TODO:从文件中获取observation，如果要修改输入的话可能要在这里修改
    """
    🧠 总结一句话

这个函数做了四件核心事情：

1️⃣ 从文件中读 traffic matrix
2️⃣ 删除“节点到自身”的流量
3️⃣ 把每个 demand 扩展成多个 path 节点特征
4️⃣ 拼接链路容量，形成神经网络输入状态
    """
    def _read_obs(self):
        """Return observation (capacity + traffic matrix) from files."""

        topo, topo_fname, tm_fname = self.problems[self.idx]
        #价值流量矩阵
        with open(tm_fname, 'rb') as f:
            tm = pickle.load(f)
        # remove demands within nodes
        tm = torch.FloatTensor(
            [[ele]*self.num_path for i, ele in enumerate(tm.flatten())
                if i % len(tm) != i//len(tm)]).flatten()# i % len(tm) != i//len(tm)排除对角线，即自己到自己的流量
        obs = torch.concat([self.capacity, tm]).to(self.device)
        # simulate link failures in testing
        if self.num_failure > 0 and self.idx_start == self.test_start:
            idx_failure = torch.tensor(
                random.sample(range(self.num_edge_node),
                self.num_failure)).to(self.device)
            obs[idx_failure] = 0
        # 最终 obs 是一个一维 tensor：
        #
        # obs.shape = (num_edge_node + num_path_node,)
        #
        #
        # 内容结构：
        #
        # [capacity_e1, capacity_e2, ..., demand_p1, demand_p2, demand_p3, ...]
        return obs

    def _next_obs(self):
        """Return next observation (capacity + traffic matrix)."""

        self.idx += 1
        if self.idx == self.idx_stop:
            self.idx = self.idx_start
        self.obs = self._read_obs()
        return self.obs

    def render(self):
        """Return a dictionary for the details of the current problem"""

        topo, topo_fname, tm_fname = self.problems[self.idx]
        problem_dict = {
            'problem_name': topo,
            'obj': self.obj,
            'tm_fname': tm_fname.split('/')[-1],
            'num_node': self.G.number_of_nodes(),
            'num_edge': self.G.number_of_edges(),
            'num_path': self.num_path,
            'edge_disjoint': self.edge_disjoint,
            'dist_metric': self.dist_metric,
            'traffic_model': tm_fname.split('/')[-2],
            'traffic_seed': int(tm_fname.split('_')[-3]),
            'scale_factor': float(tm_fname.split('_')[-2]),
            'total_demand': self.obs[
                -self.num_path_node::self.num_path].sum().item(),
        }
        return problem_dict
#计算奖励
    def step(self, raw_action, num_sample=0, num_admm_step=0):
        """Return the reward of current action.

        Args:
            raw_action: raw action from actor
            num_sample: number of samples for reward during training
            num_admm_step: number of ADMM steps during testing
        """
        failed_link=self.failed_link
        info = {}
        if self.idx_start == self.train_start:# 训练阶段
            reward = self.take_action(raw_action, num_sample)
        else: #测试阶段
            start_time = time.time()
            action = self.transform_raw_action(raw_action)
            if self.obj == 'total_flow':
                # total flow require no constraint violation
                action = self.ADMM.tune_action(self.obs, action, num_admm_step)
                action = self.round_action(action)
            info['runtime'] = time.time() - start_time
            info['sol_mat'] = self.extract_sol_mat(action)
            #TODO:在这里实现reweave的局部路径修补机制
            #print("-----------------------------")
            #print(action.size())
            if len(failed_link)==0:
                #没有链路故障
                reward = self.get_obj(action)
            else:
                #有链路故障
                #reward = self.get_obj_with_failure_without_local_backup(action,failed_link)
                reward = self.get_obj_with_failure_by_local_backup(action, failed_link)
        # next observation
        self._next_obs()
        return reward, info
# 得到目标结果
    def get_obj(self, action):
        """Return objective."""

        if self.obj == 'total_flow':
            return action.sum(axis=-1)
        elif self.obj == 'min_max_link_util':
            return (torch_scatter.scatter(
                action[self.p2e[0]], self.p2e[1]
                )/self.obs[:-self.num_path_node]).max()
# 有局部路径的情况下进行测试
    def get_obj_with_failure_by_local_backup(self,action, failed_link):
        """
        处理故障链路后计算目标指标（总流量/MLU）
        :param action: 一维Tensor，长度=路径总数，每个值是对应路径的分配流量
        :param failed_link: list[tuple]，如 [(0,1), (1,2)]，表示故障的链路（节点对）
        :return: 目标指标值（总流量/MLU）
        """
        #1.故障链路预处理
        failed_edge_ids = []  # 故障链路的编号
        # 获取原始网络容量
        #capacities = self.obs[:-self.num_path_node].clone()
        for (u,v) in failed_link:
            if (u, v) in self.edge2idx_dict:
                failed_edge_ids.append(self.edge2idx_dict[(u, v)])

        if not failed_edge_ids:
            #没有有效故障链路
            return self.get_obj(action)
        #2.找到受故障影响的路径
        # 使用 torch_scatter 把路径流量(action)聚合到链路(edge)上
        # dim_size 保证生成的 tensor 长度等于 edge 的总数
        edge_flow = torch_scatter.scatter(
                action[self.p2e[0]],
                self.p2e[1],
                dim_size=self.num_edge_node
            ).clone()  # 必须 clone，因为后续要在上面加减重分配的流量
        lost_flow = 0.0  # 记录因为找不到备份路径而丢失的流量

        #3.流量重分配（故障路径→备份路径）
        for (u,v) in failed_link:
            edge_id=self.edge2idx_dict[(u,v)]
            #该故障链路上的流量
            affected_flow=edge_flow[edge_id].item()
            if affected_flow<=0:
                continue
            edge_flow[edge_id] = 0.0
            # 取出备份路径
            backup_paths=self.local_backup_paths[(u,v)]
            valid_backup_paths = [] #可以使用的备份路径
            if not backup_paths:
                # 如果完全没有备份路径，这部分流量宣告丢失
                lost_flow += affected_flow
            else:
                #检查备份路径是否存在故障链路
                #len_backup_num==len(backup_paths)
                for backup_path in backup_paths:
                    if not any(edge in backup_path for edge in failed_link):
                        #有故障链路，无法使用该备份路径
                        valid_backup_paths.append(backup_path)

                # 将该故障链路上的流量均分到多条备份路径上 (默认采取均分)
                if len(valid_backup_paths) == 0:
                    lost_flow+=affected_flow
                else:

                    spilt_flow=affected_flow/len(valid_backup_paths)
            #对备用路径上的流量进行分配
            for backup_path in valid_backup_paths:
                for edge in backup_path:
                    edge_idx=self.edge2idx_dict[edge]
                    edge_flow[edge_idx]+=spilt_flow


        #4.用new_action重新计算目标指标
        if self.obj == 'total_flow':
            #return torch.tensor(action.sum().item()-lost_flow, device=self.device)
            original_total_flow = action.sum().item()
            # 备份原始容量并排除故障链路
            capacities = self.obs[:-self.num_path_node].clone()
            capacities[failed_edge_ids] = float('inf')  # 故障链路上由于流量已经清零，无需计算溢出

            # 计算每条链路上由于备份流量涌入导致的溢出量 (edge_flow - capacity，如果没超载则是 0)
            overflow_per_edge = (edge_flow - capacities).relu()
            # 全网总溢出量 (这部分流量会在路由器因为队列塞满而丢弃)
            total_overflow = overflow_per_edge.sum().item()
            print("total_overflow: {:.2f}".format(total_overflow))
            # 实际成功投递的流量 = 原总流量 - 没有备用路径丢失的 - 备用路径拥塞丢弃的
            # 使用 max 防止多重拥塞导致扣减出负数
            final_delivered_flow = max(0.0, original_total_flow - lost_flow - total_overflow)

            return torch.tensor(final_delivered_flow, device=self.device)

        elif self.obj == 'min_max_link_util':


            # 把故障链路的容量设为正无穷，避免计算利用率时出现 0/0 或抛出除零异常
            # 因为它们的 edge_flow 已经被设为 0 了，0 / inf = 0，不影响全局 Max
            capacities = self.obs[:-self.num_path_node].clone()
            capacities[failed_edge_ids] = float('inf')

            # 计算重分配后的最大链路利用率 (Max Link Utilization)
            mlu = (edge_flow / capacities).max()
            return mlu

#没有局部路径的时候，端到端重新分配（对比实验）
    def get_obj_with_failure_without_local_backup(self, action,failed_link):
        # 1. 提取故障链路的全局编号
        failed_edge_ids = []
        for (u, v) in failed_link:
            if (u, v) in self.edge2idx_dict:
                failed_edge_ids.append(self.edge2idx_dict[(u, v)])

        if not failed_edge_ids:
            return self.get_obj(action)
            # 2. 找出所有受影响瘫痪的“端到端路径”
        failed_edge_tensor = torch.tensor(failed_edge_ids, dtype=torch.long, device=self.device)
        is_edge_failed = torch.zeros(self.num_edge_node, dtype=torch.bool, device=self.device)
        is_edge_failed[failed_edge_tensor] = True
        #每条路径上所有链路的故障标记加起来
        edge_fails_on_path=is_edge_failed[self.p2e[1]].int() # 得到一个一维向量[0,1,0...]表示这个位置的path对应的链路有故障
        path_fails_sum = torch_scatter.scatter(
            edge_fails_on_path,
            self.p2e[0],
            dim_size=self.num_path_node,
            reduce="sum"
        )
        is_path_failed = path_fails_sum > 0  # 一维 bool 张量，True表示路径瘫痪
        # 3. 按Demand (源目节点对的数量)进行端到端流量重分配
        # 将一维流量分配变形为二维：(Demand数量, 每对Demand的路径数)，方便同组内转移流量
        action_2d = action.clone().reshape(-1, self.num_path)
        is_path_failed_2d = is_path_failed.reshape(-1, self.num_path) # 0代表这个path故障，(Demand数量, 每对Demand的路径数)
        is_path_healthy_2d = ~is_path_failed_2d
        # 计算每个 Demand 有多少流量因为故障要重新分配
        flow_to_reallocate = (action_2d * is_path_failed_2d).sum(dim=1)  # 形状: (Demand数量,)
        # 将瘫痪路径上的流量清零
        action_2d[is_path_failed_2d] = 0.0
        # 获取健康路径的当前分配情况，作为重新分配的比例权重
        remaining_flow = action_2d.sum(dim=1)  # 每个 Demand 在健康路径上剩余的总流量，形状: (Demand数量,)
        num_healthy_paths = is_path_healthy_2d.sum(dim=1)  # 每个 Demand 还剩几条健康路径，形状: (Demand数量,)
        # case1:如果健康路径上本来就有流量，按原有的流量比例进行同比例放大
        # 构建重分配权重矩阵
        weights = torch.zeros_like(action_2d)
        case1 = remaining_flow > 0
        if case1.any():
            # case1是1维，代表第几个demand,weights是2维，行是Demand数量，列是每对Demand的路径数，值是每条路径的分流权重
            # 通过广播机制，可以得到每个Demand对应每条路径的分流权重
            #每个位置的值是通过该path原来的流量除以健康路径的流量得到的，表示把原来所有流量分配到健康路径上的比例，如果这个路径有故障，前面已经把对应action_2d的值设为0了
            weights[case1] = action_2d[case1] / remaining_flow[case1].unsqueeze(1)

        # case2：如果健康路径原本分配的流量是0（比如全压在主路径上了），那就把丢失的流量均分给剩下的健康路径
        case2 = (remaining_flow == 0) & (num_healthy_paths > 0)
        if case2.any():
            #1/剩余健康路径数   0/剩余健康路径数（对故障path）
            weights[case2] = is_path_healthy_2d[case2].float() / num_healthy_paths[case2].unsqueeze(1).float()
        # 把重新分配的流量乘以权重，重新加给健康路径
        action_2d += flow_to_reallocate.unsqueeze(1) * weights
        # 重新展平为一维张量，这就是端到端重路由后的新动作分配
        new_action = action_2d.flatten()
        # 4. 用重新分配后的路径流量 (new_action) 计算目标指标
        if self.obj == 'total_flow':
            #return new_action.sum()
            # 备份原始容量
            original_capacities = self.obs[:-self.num_path_node].clone()
            # 临时将故障链路容量设为 0
            self.obs[:-self.num_path_node][failed_edge_tensor] = 1e-5
            #  round_action 进行路径级别的削减 (模拟丢包)
            valid_new_action = self.round_action(new_action, round_demand=False, round_capacity=True)
            # 恢复环境原始容量
            self.obs[:-self.num_path_node] = original_capacities

            return valid_new_action.sum()
        elif self.obj == 'min_max_link_util':
            # 用新的 action 重新计算链路层面的流量
            new_edge_flow = torch_scatter.scatter(
                new_action[self.p2e[0]],
                self.p2e[1],
                dim_size=self.num_edge_node
            )
            capacities = self.obs[:-self.num_path_node].clone()
            # 同样要把故障链路的容量设为正无穷，屏蔽它们对利用率最大值的干扰
            capacities[failed_edge_tensor] = float('inf')

            return (new_edge_flow / capacities).max()
# 把神经网络输入的结果（raw action)转换为流量分配方案
    def transform_raw_action(self, raw_action):
        """Return network flow allocation as action.

        Args:
            raw_action: raw action directly from ML output
        """
        # clamp raw action between raw_action_min and raw_action_max
        raw_action = torch.clamp(
            raw_action, min=self.raw_action_min, max=self.raw_action_max)

        # translate ML output to split ratio through softmax
        # 1 in softmax represent unallocated traffic
        ## 第一步：对原始动作做指数运算（保证所有值为正）
        raw_action = raw_action.exp()
        #最终效果：所有路径的分配比例之和 = sum(指数值) / (1+sum(指数值)) < 1，剩余的 1/(1+sum(指数值)) 是未分配流量
        raw_action = raw_action/(1+raw_action.sum(axis=-1)[:, None])

        # translate split ratio to flow
        ## 展平为一维张量（消除多维结构）
        # 乘以各路径的容量/需求基数，得到实际流量
        raw_action = raw_action.flatten() * self.obs[-self.num_path_node:]

        return raw_action

    def round_action(
            self, action, round_demand=True, round_capacity=True,
            num_round_iter=2):
        """Return rounded action.
        Action can still violate constraints even after ADMM fine-tuning.
        This function rounds the action through cutting flow.

        Args:
            action: input action
            round_demand: whether to round action for demand constraints
            round_capacity: whether to round action for capacity constraints
            num_round_iter: number of rounds when iteratively cutting flow
        """

        demand = self.obs[-self.num_path_node::self.num_path]
        capacity = self.obs[:-self.num_path_node]

        # reduce action proportionally if action exceed demand
        if round_demand:
            action = action.reshape(-1, self.num_path)
            ratio = action.sum(-1) / demand
            action[ratio > 1, :] /= ratio[ratio > 1, None]
            action = action.flatten()

        # iteratively reduce action proportionally if action exceed capacity
        if round_capacity:
            path_flow = action
            path_flow_allocated_total = torch.zeros(path_flow.shape)\
                .to(self.device)
            for round_iter in range(num_round_iter):
                # flow on each edge
                edge_flow = torch_scatter.scatter(
                    path_flow[self.p2e[0]], self.p2e[1])
                # util of each edge
                util = 1 + (edge_flow/capacity-1).relu()
                # propotionally cut path flow by max util
                util = torch_scatter.scatter(
                    util[self.p2e[1]], self.p2e[0], reduce="max")
                path_flow_allocated = path_flow/util
                # update total allocation, residual capacity, residual flow
                path_flow_allocated_total += path_flow_allocated
                if round_iter != num_round_iter - 1:
                    capacity = (capacity - torch_scatter.scatter(
                        path_flow_allocated[self.p2e[0]], self.p2e[1])).relu()
                    path_flow = path_flow - path_flow_allocated
            action = path_flow_allocated_total

        return action

    def take_action(self, raw_action, num_sample):
        '''Return an approximate reward for action for each node pair.
        To make function fast and scalable on GPU, we only calculate delta.
        We assume when changing action in one node pair:
        (1) The change in edge utilization is very small;
        (2) The bottleneck edge in a path does not change due to (1).
        For evary path after change:
            path_flow/max(util, 1) =>
            (path_flow+delta_path_flow)/max(util+delta_util, 1)
            if util < 1:
                reward = - delta_path_flow
            if util > 1:
                reward = - delta_path_flow/(util+delta_util)
                    + path_flow*delta_util/(util+delta_util)/util
                    approx delta_path_flow/util - path_flow/util^2*delta_util

        Args:
            raw_action: raw action from policy network
            num_sample: number of samples in estimating reward
        '''

        path_flow = self.transform_raw_action(raw_action)
        edge_flow = torch_scatter.scatter(path_flow[self.p2e[0]], self.p2e[1])
        util = edge_flow/self.obs[:-self.num_path_node]

        # sample from uniform distribution [mean_min, min_max]
        distribution = Uniform(
            torch.ones(raw_action.shape).to(self.device)*self.raw_action_min,
            torch.ones(raw_action.shape).to(self.device)*self.raw_action_max)
        reward = torch.zeros(self.num_path_node//self.num_path).to(self.device)

        if self.obj == 'total_flow':

            # find bottlenack edge for each path
            util, path_bottleneck = torch_scatter.scatter_max(
                util[self.p2e[1]], self.p2e[0])
            path_bottleneck = self.p2e[1][path_bottleneck]

            # prepare -path_flow/util^2 for reward
            coef = path_flow/util**2
            coef[util < 1] = 0
            coef = torch_scatter.scatter(
                coef, path_bottleneck).reshape(-1, 1)

            # prepare path_util to bottleneck edge_util
            bottleneck_p2e = torch.sparse_coo_tensor(
                self.p2e, (1/self.obs[:-self.num_path_node])[self.p2e[1]],
                [self.num_path_node, self.num_edge_node])

            # sample raw_actions and change each node pair at a time for reward
            for _ in range(num_sample):
                sample = distribution.rsample()

                # add -delta_path_flow if util < 1 else -delta_path_flow/util
                delta_path_flow = self.transform_raw_action(sample) - path_flow
                reward += -(delta_path_flow/(1+(util-1).relu()))\
                    .reshape(-1, self.num_path).sum(-1)

                # add path_flow/util^2*delta_util for each path
                delta_path_flow = torch.sparse_coo_tensor(
                    torch.stack(
                        [torch.arange(self.num_path_node//self.num_path)
                            .to(self.device).repeat_interleave(self.num_path),
                            torch.arange(self.num_path_node).to(self.device)]),
                    delta_path_flow,
                    [self.num_path_node//self.num_path, self.num_path_node])
                # get utilization changes on edge
                # do not use torch_sparse.spspmm()
                # "an illegal memory access was encountered" in large topology
                delta_util = torch.sparse.mm(delta_path_flow, bottleneck_p2e)
                reward += torch.sparse.mm(delta_util, coef).flatten()

        elif self.obj == 'min_max_link_util':

            # find link with max utilization
            max_util_edge = util.argmax()

            # prepare paths related to max_util_edge
            max_util_paths = torch.zeros(self.num_path_node).to(self.device)
            max_util_paths[self.p2e[0, self.p2e[1] == max_util_edge]] =\
                1/self.obs[max_util_edge]

            # sample raw_actions and change each node pair at a time for reward
            for _ in range(num_sample):
                sample = distribution.rsample()

                delta_path_flow = self.transform_raw_action(sample) - path_flow
                delta_path_flow = torch.sparse_coo_tensor(
                    torch.stack(
                        [torch.arange(self.num_path_node//self.num_path)
                            .to(self.device).repeat_interleave(self.num_path),
                            torch.arange(self.num_path_node).to(self.device)]),
                    delta_path_flow,
                    [self.num_path_node//self.num_path, self.num_path_node])
                reward += torch.sparse.mm(
                    delta_path_flow, max_util_paths.reshape(-1, 1)).flatten()

        return reward/num_sample

    def _read_graph_json(self, topo):
        """Return network topo from json file."""

        assert topo.endswith(".json")
        with open(os.path.join(TOPOLOGIES_DIR, topo)) as f:
            data = json.load(f)
        return json_graph.node_link_graph(data)
# TODO 需要设计一下获得不同时间片的路径文件
    #返回路径文件名
    def path_full_fname(self, topo, num_path, edge_disjoint, dist_metric):
        """Return full name of the topology path."""

        return os.path.join(
            TOPOLOGIES_DIR, "paths", "path-form",
            "{}-{}-paths_edge-disjoint-{}_dist-metric-{}-dict.pkl".format(
                topo, num_path, edge_disjoint, dist_metric))
# TODO 设计一下获得不同时间片的路径
    #读取路径文件获得路径
    def get_path(self, topo, num_path, edge_disjoint, dist_metric):
        """Return path dictionary."""

        self.path_fname = self.path_full_fname(
            topo, num_path, edge_disjoint, dist_metric)
        print("Loading paths from pickle file", self.path_fname)
        try:
            with open(self.path_fname, 'rb') as f:
                path_dict = pickle.load(f)
                print("path_dict size:", len(path_dict))
                return path_dict
        except FileNotFoundError:
            print("Creating paths {}".format(self.path_fname))
            path_dict = self.compute_path(
                topo, num_path, edge_disjoint, dist_metric)
            print("Saving paths to pickle file")
            with open(self.path_fname, "wb") as w:
                pickle.dump(path_dict, w)
        return path_dict
# 如果没有读到路径文件，则计算路径
    def compute_path(self, topo, num_path, edge_disjoint, dist_metric):
        """Return path dictionary through computation."""

        path_dict = {}
        G = graph_copy_with_edge_weights(self.G, dist_metric)
        for s_k in G.nodes:
            for t_k in G.nodes:
                if s_k == t_k:
                    continue
                paths = find_paths(G, s_k, t_k, num_path, edge_disjoint)
                paths_no_cycles = [remove_cycles(path) for path in paths]
                path_dict[(s_k, t_k)] = paths_no_cycles
        return path_dict

    def get_regular_path(self, topo, num_path, edge_disjoint, dist_metric):
        """Return path dictionary with the same number of paths per demand.
        Fill with the first path when number of paths is not enough.
        """

        path_dict = self.get_path(topo, num_path, edge_disjoint, dist_metric)
        for (s_k, t_k) in path_dict:
            ## 路径数量不足：用第一条路径重复填充，补到目标数量
            if len(path_dict[(s_k, t_k)]) < self.num_path:
                path_dict[(s_k, t_k)] = [
                    path_dict[(s_k, t_k)][0] for _
                    in range(self.num_path - len(path_dict[(s_k, t_k)]))]\
                    + path_dict[(s_k, t_k)]
            elif len(path_dict[(s_k, t_k)]) > self.num_path:
                path_dict[(s_k, t_k)] = path_dict[(s_k, t_k)][:self.num_path]
        return path_dict

    def get_topo_matrix(self, topo, num_path, edge_disjoint, dist_metric):
        """
        Return matrices related to topology.
        edge_index, edge_index_values: index and value for matrix
        D^(-0.5)*(adjacent)*D^(-0.5) without self-loop
        p2e: [path_node_idx, edge_nodes_inx]
        """

        # get regular path dict
        path_dict = self.get_regular_path(
            topo, num_path, edge_disjoint, dist_metric)

        # edge nodes' degree, index lookup
        #对每个链路找一个链路标识
        self.edge2idx_dict = {edge: idx for idx, edge in enumerate(self.G.edges)}
        node2degree_dict = {}
        edge_num = len(self.G.edges)

        # build edge_index
        src, dst, path_i = [], [], 0
        for s in range(len(self.G)):
            for t in range(len(self.G)):
                if s == t:
                    continue
                for path in path_dict[(s, t)]:
                    for (u, v) in zip(path[:-1], path[1:]):
                        #src.append(edge_num+path_i)：给当前路径分配一个唯一标识（edge_num 是基础值，path_i 是路径序号），存入 src；
                        src.append(edge_num+path_i)
                        dst.append(self.edge2idx_dict[(u, v)])

                        if src[-1] not in node2degree_dict:
                            node2degree_dict[src[-1]] = 0
                        node2degree_dict[src[-1]] += 1
                        if dst[-1] not in node2degree_dict:
                            node2degree_dict[dst[-1]] = 0
                        node2degree_dict[dst[-1]] += 1
                    path_i += 1

        # edge_index is D^(-0.5)*(adj)*D^(-0.5) without self-loop
        edge_index_values = torch.tensor(
            [1/math.sqrt(node2degree_dict[u]*node2degree_dict[v])
                for u, v in zip(src+dst, dst+src)]).to(self.device)
        edge_index = torch.tensor(
            [src+dst, dst+src], dtype=torch.long).to(self.device)
        p2e = torch.tensor([src, dst], dtype=torch.long).to(self.device)
        p2e[0] -= len(self.G.edges)
        """
        最终p2e的内容：
        tensor([[0, 0, 1],
        [0, 1, 2]])

        p2e[0]（路径索引）： [0 0 1]
        p2e[1]（链路索引）： [0 1 2]
        表示路径0经过链路0，链路1，路径1经过链路2
        """


        return edge_index, edge_index_values, p2e

    def extract_sol_mat(self, action):
        """return sparse solution matrix.
        Solution matrix is of dimension num_of_demand x num_of_edge.
        The i, j entry represents the traffic flow from demand i on edge j.
        """

        # 3D sparse matrix to represent which path, which demand, which edge
        sol_mat_index = torch.stack([
            self.p2e[0] % self.num_path,
            torch.div(self.p2e[0], self.num_path, rounding_mode='floor'),
            self.p2e[1]])

        # merge allocation from different paths of the same demand
        sol_mat = torch.sparse_coo_tensor(
            sol_mat_index,
            action[self.p2e[0]],
            (self.num_path,
                self.num_path_node//self.num_path,
                self.num_edge_node))
        sol_mat = torch.sparse.sum(sol_mat, [0])

        return sol_mat
