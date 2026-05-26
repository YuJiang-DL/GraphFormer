#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 22:55:06 2021

@author: kyungsub
"""

import torch
import os
import pandas as pd
import networkx as nx
import torch_geometric.utils as g_util
from torch_geometric.data import Data
import numpy as np

from sklearn.metrics import pairwise_distances

from tqdm import tqdm
from torch_geometric.transforms import LocalCartesian, Cartesian, Polar


pd.options.mode.chained_assignment = None

def false_graph_filtering(distance_thresh):
    root_dir = r' '
    origin_file_dir = r" "
    different_value = 0.75
    supernode_sample = os.listdir(root_dir)
    origin_file_list = os.listdir(origin_file_dir)

    original_file_list = []
    for file in supernode_sample:
        if os.path.isfile(os.path.join(root_dir, file)):
            original_file_list.append(file.split('_')[0])
    sample_files = list(set(original_file_list))

    supernode_files = [os.path.join(root_dir, item + '_' + str(different_value) + '.csv') for item in sample_files]

    pt_files = []
    for item in sample_files:
        if os.path.isfile(os.path.join(root_dir, item + '_' + str(different_value) + '_graph_torch_new.pt')):
            pt_files.append(os.path.join(root_dir, item + '_' + str(different_value) + '_graph_torch_new.pt'))
        else:
            if os.path.isfile(os.path.join(root_dir, item + '_' + str(different_value) + '_graph_torch.pt')):
                pt_files.append(os.path.join(root_dir, item + '_' + str(different_value) +'_graph_torch.pt'))
            else:
                pt_files.append('pass')


    with tqdm(total=len(pt_files)) as pbar:
        for sample_name, supernode_path, pt in zip(sample_files, supernode_files, pt_files):
            #print(sample_name)
            #print(supernode_path)
            #print(os.path.join(root_dir, sample_name + "_" + str(different_value) + "_graph_torch_" + str(
                #distance_thresh) + "_artifact_sophis_final.pt"))
            if not os.path.isfile(os.path.join(root_dir, sample_name + "_" + str(different_value) + "_graph_torch_" + str(
                distance_thresh) + "_artifact_sophis_final.pt")):
                #print(pt)
                if 'pass' not in pt:

                    #location = os.path.join('../Sample_data_for_demo/Graph_test/TCGA/KIRC/original/', sample_name + '_node_location_list.csv')
                    location = os.path.join(origin_file_dir,sample_name + '_node_location_list.csv')
                    # print(pt)
                    if 'graph_torch_new' in pt:
                        graph = Data.from_dict(torch.load(pt))
                    else:
                        graph = torch.load(pt)

                    if os.path.isfile(supernode_path):

                        supernode = pd.read_csv(supernode_path)
                        location = pd.read_csv(location)

                        sample = sample_name.split('_')[0]

                        supernode_x = []
                        supernode_y = []
                        supernode_num = supernode['Unnamed: 0'].tolist()
                        # for _node in range(graph.num_nodes):
                        #    supernode_num.append(supernode.loc[_node][0])

                        supernode_x = location.loc[supernode_num]['X']
                        supernode_y = location.loc[supernode_num]['Y']
                        # supernode_x.append(node_x)
                        # supernode_y.append(node_y)
                        coordinate_df = pd.DataFrame({'X': supernode_x, 'Y': supernode_y})
                        coordinate_list = np.array(coordinate_df.values.tolist())
                        coordinate_matrix = pairwise_distances(coordinate_list, n_jobs=8)
                        adj_matrix = np.where(coordinate_matrix >= distance_thresh, 0, 1)
                        Edge_label = np.where(adj_matrix == 1)

                        # 检测并剔除孤立节点（只连接一个邻居的）
                        # 统计那些「既没有入边也没有出边」或者「只连了一个点」的孤立节点，用于后续筛除。
                        Adj_from = np.unique(Edge_label[0], return_counts=True)
                        Adj_to = np.unique(Edge_label[1], return_counts=True)

                        Adj_from_singleton = Adj_from[0][Adj_from[1] == 1]
                        Adj_to_singleton = Adj_to[0][Adj_to[1] == 1]

                        Adj_singleton = np.intersect1d(Adj_from_singleton, Adj_to_singleton)

                        coordinate_matrix_modify = coordinate_matrix

                        fromlist = Edge_label[0].tolist()
                        tolist = Edge_label[1].tolist()
                        # 构建 PyTorch Geometric 的新边 edge_index
                        # 如果两次阈值相同，理论上边不变
                        edge_index = torch.tensor([fromlist, tolist], dtype=torch.long)
                        graph.edge_index = edge_index

                        connected_graph = g_util.to_networkx(graph, to_undirected=True)

                        # visualize_graph(connected_graph, color=graph.y)

                        # nx.connected_components是NetworkX库中的一个函数，用于查找无向图中的连通分量。
                        # 它返回一个生成器对象，其中每个元素都是一个集合，表示一个连通分量中的所有节点。
                        # 连通分量指的是无向图中的一个子图，其中任意两个节点都可以通过路径相互到达，而与该子图外的其他节点不连通。
                        # 一个无向图可能包含多个连通分量。在连通分量中，任意两个节点都存在路径相互可达，因此连通分量是图的一个重要概念。
                        # for i in nx.connected_components(connected_graph):
                        #     # print(len(i))  # 求图个连通分量节点集合
                        #
                        #     '''
                        #     这段代码使用了NetworkX库来寻找连通图中节点数大于100的子图，
                        #     并将这些子图存入一个列表中。
                        #     具体来说，代码首先通过nx.connected_components(connected_graph)函数找到连通图中所有的连通子图，并遍历每个子图。
                        #     然后，对于每个子图，如果其节点数大于100，就将该子图作为参数传入connected_graph.subgraph(item_graph).copy()函数，
                        #     得到一个新的子图，并将其添加到结果列表中。最后，返回结果列表。
                        #     '''
                        connected_graph = [connected_graph.subgraph(item_graph).copy() for item_graph in
                                           nx.connected_components(connected_graph) if len(item_graph) > 1]
                        connected_graph_node_list = []
                        for graph_item in connected_graph:
                            connected_graph_node_list.extend(list(graph_item.nodes))
                        connected_graph = connected_graph_node_list
                        connected_graph = list(connected_graph)
                        new_node_order_dict = dict(zip(connected_graph, range(len(connected_graph))))
                        # new_node_order_dict = dict(zip(range(len(connected_graph)), connected_graph))

                        new_feature = graph.x[connected_graph]
                        new_edge_index = graph.edge_index.numpy()
                        new_edge_mask_from = np.isin(new_edge_index[0], connected_graph)
                        new_edge_mask_to = np.isin(new_edge_index[1], connected_graph)
                        new_edge_mask = new_edge_mask_from * new_edge_mask_to
                        new_edge_index_from = new_edge_index[0]
                        new_edge_index_from = new_edge_index_from[new_edge_mask]
                        new_edge_index_from = [new_node_order_dict[item] for item in new_edge_index_from]
                        new_edge_index_to = new_edge_index[1]
                        new_edge_index_to = new_edge_index_to[new_edge_mask]
                        new_edge_index_to = [new_node_order_dict[item] for item in new_edge_index_to]

                        new_edge_index = torch.tensor([new_edge_index_from, new_edge_index_to], dtype=torch.long)

                        new_supernode = supernode.iloc[connected_graph]
                        new_supernode = new_supernode.reset_index()
                        # new_supernode = new_supernode.reindex([item[1] for item in new_node_order_dict.items()])
                        new_supernode.to_csv(supernode_path.split('.csv')[0] + '_' + str(distance_thresh) + '_artifact_sophis_final.csv')

                        actual_pos = location.iloc[new_supernode['Unnamed: 0']]
                        actual_pos = actual_pos[['X', 'Y']].to_numpy()
                        actual_pos = torch.tensor(actual_pos)
                        actual_pos = actual_pos.float()

                        pos_transfrom = Polar()
                        new_graph = Data(x=new_feature, edge_index=new_edge_index, pos=actual_pos * 256.0)
                        # print(new_graph)
                        # print(supernode_path)
                        try:
                            # 尝试调用 pos_transfrom 转换图数据
                            new_graph = pos_transfrom(new_graph)
                        except RuntimeError as e:
                            # 捕获 RuntimeError 并跳过该图，继续下一个
                            print(f"警告: 处理图时发生错误，错误信息: {e}. 跳过此图。")
                            continue  # 跳过当前图，继续下一个图的处理
                        # new_graph = pos_transfrom(new_graph)


                        torch.save(new_graph, os.path.join(root_dir, sample_name + "_" + str(different_value) +"_graph_torch_" + str(
                            distance_thresh) + "_artifact_sophis_final.pt"))

            pbar.update()


# false_graph_filtering(4.3)