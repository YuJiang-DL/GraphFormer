#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 21:10:37 2021

@author: kyungsub
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import torch
OPENSLIDE_PATH = r' '
import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide as osd
else:
    import openslide as osd
from torchvision import transforms
from torch_geometric.data import Data
from patch_construction.EfficientNet import EfficientNet
from graph_construction import false_graph_filtering
from skimage.filters import threshold_multiotsu
import pickle
import argparse
from openslide import OpenSlideError
class SurvivalImageDataset():

    """
    Target dataset has the list of images such as
    _patientID_SurvDay_Censor_TumorStage_WSIPos.tif
    """

    def __init__(self, image, x, y, transform):

        self.image = image
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len((self.image))

    def __getitem__(self, idx):

        """
        patientID, SurvivalDuration, SurvivalCensor, Stage,
        ProgressionDuration, ProgressionCensor, MetaDuration, MetaCensor
        """
        transform = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
                ])
        #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        image = self.image[idx]
        x = self.x[idx]
        y = self.y[idx]
        image = image.convert('RGB')
        R = transform(image)

        sample = { 'image' : R,'X' : torch.tensor(x), 'Y' : torch.tensor(y) }

        return sample



def supernode_generation(image, model_ft, device, Argument, save_dir):

    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    origin_dir = os.path.join(save_dir, 'original')
    print(origin_dir)
    if os.path.exists(origin_dir) is False:
        os.mkdir(origin_dir)

    superpatch_dir = os.path.join(save_dir, 'superpatch')
    print(superpatch_dir)
    if os.path.exists(superpatch_dir) is False:
        os.mkdir(superpatch_dir)

    transform = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
                ])

    threshold = Argument.threshold
    spatial_threshold = Argument.spatial_threshold

    sample = image.split('/')[-1].split('.')[0]
    print(sample)

    image_path = image
    try:
        slideimage = osd.OpenSlide(image_path)
    except:
        print('openslide error')
        return 0
    downsampling = slideimage.level_downsamples
    # print(len(downsampling))
    if len(downsampling) > 1:
        best_downsampling_level = 1
        downsampling_factor = int(slideimage.level_downsamples[best_downsampling_level])

        # Get the image at the requested scale
        svs_native_levelimg = slideimage.read_region((0, 0), best_downsampling_level, slideimage.level_dimensions[best_downsampling_level])
        svs_native_levelimg = svs_native_levelimg.convert('L')
        img = np.array(svs_native_levelimg)

        thresholds = threshold_multiotsu(img)
        regions = np.digitize(img, bins=thresholds)
        regions[regions == 1] = 0
        regions[regions == 2] = 1
        thresh_otsu = regions # 1是背景

        imagesize = Argument.imagesize
        downsampled_size = int(imagesize /downsampling_factor)
        Width = slideimage.dimensions[0]
        Height = slideimage.dimensions[1]
        num_row = int(Height/imagesize) + 1
        num_col = int(Width/imagesize) + 1
        x_list = []
        y_list = []
        feature_list = []
        x_y_list = []
        counter = 0
        inside_counter = 0
        temp_patch_list = []
        temp_x = []
        temp_y = []

        with tqdm(total = num_row * num_col) as pbar_image:
            for i in range(0, num_col):
                for j in range(0, num_row):

                    if thresh_otsu.shape[1] >= (i+1)*downsampled_size:
                        if thresh_otsu.shape[0] >= (j+1)*downsampled_size:
                            cut_thresh = thresh_otsu[j*downsampled_size:(j+1)*downsampled_size, i*downsampled_size:(i+1)*downsampled_size]
                        else:
                            cut_thresh = thresh_otsu[(j)*downsampled_size:thresh_otsu.shape[0], i*downsampled_size:(i+1)*downsampled_size]
                    else:
                        if thresh_otsu.shape[0] >= (j+1)*downsampled_size:
                            cut_thresh = thresh_otsu[j*downsampled_size:(j+1)*downsampled_size, (i)*downsampled_size:thresh_otsu.shape[1]]
                        else:
                            cut_thresh = thresh_otsu[(j)*downsampled_size:thresh_otsu.shape[0], (i)*downsampled_size:thresh_otsu.shape[1]]

                    if np.mean(cut_thresh) > 0.75:
                        pbar_image.update()
                        pass
                    else:

                        filter_location = (i*imagesize, j*imagesize)
                        level = 0
                        patch_size = (imagesize, imagesize)
                        location = (filter_location[0], filter_location[1])

                        # CutImage = slideimage.read_region(location, level, patch_size)
                        try:
                            CutImage = slideimage.read_region(location, level, patch_size)  # RGBA PIL.Image
                        except OpenSlideError as e:
                            print(f"[WARN] read_region failed, skip. "
                                  f"loc={location}, level={level}, patch_size={patch_size}, err={e}")
                            continue
                        except Exception as e:
                            print(f"[WARN] unexpected read error, skip. "
                                  f"loc={location}, level={level}, patch_size={patch_size}, err={e}")
                            continue

                        temp_patch_list.append(CutImage)
                        x_list.append(i)
                        y_list.append(j)
                        temp_x.append(i)
                        temp_y.append(j)
                        counter += 1
                        batchsize = 64

                        if counter == batchsize:

                            Dataset = SurvivalImageDataset(temp_patch_list, temp_x, temp_y, transform)
                            dataloader = torch.utils.data.DataLoader(Dataset,batch_size=batchsize,num_workers=0,drop_last=False)
                            for sample_img in dataloader:
                                images = sample_img['image']
                                images = images.to(device)
                                with torch.set_grad_enabled(False):
                                     classifier, features = model_ft(images)

                            if inside_counter == 0:
                                feature_list = np.concatenate((features.cpu().detach().numpy(),
                                                               classifier.cpu().detach().numpy()), axis=1)
                                temp_x = np.reshape(np.array(temp_x), (len(temp_x),1))
                                temp_y = np.reshape(np.array(temp_y), (len(temp_x),1))

                                x_y_list = np.concatenate((temp_x,temp_y),axis=1)
                            else:
                                feature_list = np.concatenate((feature_list,
                                                               np.concatenate((features.cpu().detach().numpy(),
                                                                             classifier.cpu().detach().numpy()),axis=1)), axis=0)
                                temp_x = np.reshape(np.array(temp_x), (len(temp_x),1))
                                temp_y = np.reshape(np.array(temp_y), (len(temp_x),1))

                                x_y_list = np.concatenate((x_y_list,
                                                           np.concatenate((temp_x,temp_y),axis=1)), axis=0)
                            inside_counter += 1
                            temp_patch_list = []
                            temp_x = []
                            temp_y = []
                            counter = 0

                        pbar_image.update()

            if counter < batchsize and counter >0:
                Dataset = SurvivalImageDataset(temp_patch_list, temp_x, temp_y, transform)
                dataloader = torch.utils.data.DataLoader(Dataset,batch_size=batchsize,num_workers=0,drop_last=False)
                for sample_img in dataloader:
                    images = sample_img['image']
                    images = images.to(device)
                    with torch.set_grad_enabled(False):
                         classifier, features = model_ft(images)

                    feature_list = np.concatenate((feature_list,
                                                   np.concatenate((features.cpu().detach().numpy(),
                                                                 classifier.cpu().detach().numpy()),axis=1)), axis=0)
                    temp_x = np.reshape(np.array(temp_x), (len(temp_x),1))
                    temp_y = np.reshape(np.array(temp_y), (len(temp_x),1))

                    x_y_list = np.concatenate((x_y_list,
                                               np.concatenate((temp_x,temp_y),axis=1)), axis=0)
                temp_patch_list = []
                temp_x = []
                temp_y = []
                counter = 0

        feature_df = pd.DataFrame.from_dict(feature_list)# 整张WSI的特征
        """
        每一行是一个 patch 的特征向量（1794维）
        形状：[N_patch, feature_dim]
        """
        coordinate_df = pd.DataFrame({'X': x_y_list[:,0],'Y': x_y_list[:,1]})
        """
        x_y_list 存的是每个 patch 的位置索引（不是像素坐标，而是 patch 网格位置）
        例如 (5, 10) 代表第 5 列，第 10 行的 patch
        形状：[N_patch, 2]
        """
        graph_dataframe = pd.concat([coordinate_df, feature_df], axis = 1)
        graph_dataframe = graph_dataframe.sort_values(by = ['Y', 'X'])
        graph_dataframe = graph_dataframe.reset_index(drop = True)
        coordinate_df = graph_dataframe.iloc[:,0:2] # 相比于之前的 进行了一次Y\X排序
        feature_df.to_csv(os.path.join(origin_dir, sample + '_feature_list.csv'))
        print("-------------------:", origin_dir)
        coordinate_df.to_csv(os.path.join(origin_dir, sample+'_node_location_list.csv'))
        index = list(graph_dataframe.index)
        graph_dataframe.insert(0,'index_orig', index)

        node_dict = {} # 也就是图的邻接表（Adjacency List）结构，用于描述 patch（节点）之间的连接关系。

        for i in range(len(coordinate_df)):
            node_dict.setdefault(i,[])
        # 这是所有 patch 坐标中最大的横向（X）和纵向（Y）网格坐标，用于计算网格的划分尺度。
        X = max(set(np.squeeze(graph_dataframe.loc[:, ['X']].values,axis = 1)))
        Y = max(set(np.squeeze(graph_dataframe.loc[:, ['Y']].values, axis = 1)))
        del feature_df

        # 将整张图分成 (gridNum+2) × (gridNum+2) 个区域（添加了边缘），用于分区构图（局部建边）。
        # 每个子图格子的大小就是 (X_size, Y_size)
        gridNum = 4
        X_size = int(X / gridNum)
        Y_size = int(Y / gridNum)


        # 通过 滑动网格窗口 + 条件筛选语句 对 graph_dataframe 中的 patch 进行空间坐标过滤，筛选出处于当前小格子区域内的 patch。
        with tqdm(total=(gridNum+2)*(gridNum+2)) as pbar:
            for p in range(gridNum+2):
                for q in range(gridNum+2):
                    if p == 0 :
                        if q == 0:
                            is_X = graph_dataframe['X'] <= X_size * (p+1)
                            is_X2 = graph_dataframe['X'] >= 0
                            is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                            is_Y2 = graph_dataframe['Y'] >= 0
                            X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]

                        elif q == (gridNum+1):
                            is_X = graph_dataframe['X'] <= X_size * (p+1)
                            is_X2 = graph_dataframe['X'] >= 0
                            is_Y = graph_dataframe['Y'] <= Y
                            is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                            X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]

                        else:
                            is_X = graph_dataframe['X'] <= X_size * (p+1)
                            is_X2 = graph_dataframe['X'] >= 0
                            is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                            is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                            X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                    elif p == (gridNum+1) :
                        if q == 0:
                            is_X = graph_dataframe['X'] <= X
                            is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                            is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                            is_Y2 = graph_dataframe['Y'] >= 0
                            X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                        elif q == (gridNum+1):
                            is_X = graph_dataframe['X'] <= X
                            is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                            is_Y = graph_dataframe['Y'] <= Y
                            is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                            X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                        else:
                            is_X = graph_dataframe['X'] <= X
                            is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                            is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                            is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                            X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                    else :
                        if q == 0:
                            is_X = graph_dataframe['X'] <= X_size * (p+1)
                            is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                            is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                            is_Y2 = graph_dataframe['Y'] >= 0
                            X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                        elif q == (gridNum+1):
                            is_X = graph_dataframe['X'] <= X_size * (p+1)
                            is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                            is_Y = graph_dataframe['Y'] <= Y
                            is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                            X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]
                        else:
                            is_X = graph_dataframe['X'] <= X_size * (p+1)
                            is_X2 = graph_dataframe['X'] >= (X_size *(p) - 2)
                            is_Y = graph_dataframe['Y'] <= Y_size * (q+1)
                            is_Y2 = graph_dataframe['Y'] >= (Y_size * (q) -2)
                            X_10 = graph_dataframe[is_X & is_Y & is_X2 & is_Y2]

                    if len(X_10) == 0:
                        pbar.update()
                        continue

                    coordinate_dataframe = X_10.loc[:, ['X','Y']] # 节点二维坐标，用于计算距离
                    X_10 = X_10.reset_index(drop = True)
                    coordinate_list = coordinate_dataframe.values.tolist()
                    index_list = coordinate_dataframe.index.tolist()
                    # 节点特征，用于计算相似性
                    feature_dataframe = X_10[X_10.columns.difference(['index_orig','X','Y'])]
                    feature_list = feature_dataframe.values.tolist()
                    coordinate_matrix = euclidean_distances(coordinate_list, coordinate_list)
                    coordinate_matrix = np.where(coordinate_matrix > 2.9, 0 , 1)
                    cosine_matrix = cosine_similarity(feature_list, feature_list)

                    Adj_list = (coordinate_matrix == 1).astype(int) * (cosine_matrix >= threshold).astype(int)

                    for c, item in enumerate(Adj_list):
                        for node_index in np.array(index_list)[item.astype('bool')]:
                            if node_index == index_list[c]:
                                pass
                            else:
                                node_dict[index_list[c]].append(node_index)


                    pbar.update()
        """
        graph_dataframe --> 划格子 -->
        每格筛 patch --> 计算空间欧氏距离 -->
        筛小于阈值的邻接关系 --> 
            再判断特征余弦相似度 --> 
                组合成 Adj_list --> 
                    填充 node_dict
        最终保存为 node_dict.pkl
        """
        a_file = open(os.path.join(origin_dir, sample + '_node_dict.pkl'), "wb")
        pickle.dump(node_dict, a_file)
        a_file.close()

        """
        对已有的 patch 邻接字典 node_dict 进行稀疏化和去重，以构建超节点（Supernode）
        前面已经构建了一个 patch 之间的邻接表 node_dict，
        每个节点 i 对应一个邻居列表 node_dict[i]，代表 patch i 所连接的 patch。
        但不希望所有节点都参与 supernode 构建，否则 patch 太多、结构太密、冗余严重。
        使用下面的逻辑，只保留一部分“代表性强”的 supernode，删除掉其邻居，避免冗余表示。
        """
        dict_len_list = []

        for i in range(0, len(node_dict)):
            dict_len_list.append(len(node_dict[i])) # 表示 node_dict[i] 的邻居数量（即 patch i 的“影响力”或“连接程度”）
        # 将节点按邻居数量从大到小排序
        """
        得到的是一个索引列表，邻居最多的 patch 在最前面
        优先选择连接最密集的 patch 作为 supernode 中心
        """
        arglist_strict = np.argsort(np.array(dict_len_list)) #升序排序，并返回排序后的原始索引数组（非元素值本身）。
        arglist_strict = arglist_strict[::-1] #反转，从而将排序结果从升序变为降序。

        # 去除重叠节点（即：避免重复聚合）
        """
        逻辑：
        每次取一个“最有代表性”的节点 arg_value      
        删除它的邻居节点 adj_item（从 node_dict 中删掉）
        同时从排序列表中也删除掉这些邻居，避免它们再次被选为 supernode
        最终剩下来的都是互不重叠、连接丰富的 supernode。
        """
        for arg_value in arglist_strict:
            if arg_value in node_dict.keys():
                for adj_item in node_dict[arg_value]:
                    if adj_item in node_dict.keys():
                        node_dict.pop(adj_item)
                        arglist_strict=np.delete(arglist_strict, np.argwhere(arglist_strict == adj_item))
        # 邻接列表去重（删除重复的邻接）
        for key_value in node_dict.keys():
            # set(node_dict[key_value])将列表转换为集合（set），自动去除重复元素（因为集合不允许重复值）。
            node_dict[key_value] = list(set(node_dict[key_value]))

        supernode_coordinate_x_strict = []
        supernode_coordinate_y_strict = []
        supernode_feature_strict = []
        """
        用来分别存储：
        X 坐标
        Y 坐标
        超节点的特征（聚合多个 patch 特征）
        """
        supernode_relate_value = [supernode_coordinate_x_strict,
                                   supernode_coordinate_y_strict,
                                   supernode_feature_strict]
        # 获取原始 patch 的所有特征向量
        whole_feature = graph_dataframe[graph_dataframe.columns.difference(['index_orig','X','Y'])]

        with tqdm(total = len(node_dict.keys())) as pbar_node:
            # 遍历每个 supernode, 每个 key_value 是一个 supernode 的中心 patch 索引。
            for key_value in node_dict.keys():
                 # 将这个 supernode 的空间坐标加入列表。
                supernode_relate_value[0].append(graph_dataframe['X'][key_value])
                supernode_relate_value[1].append(graph_dataframe['Y'][key_value])
                 # 聚合特征向量（自己 + 邻居）
                if len(node_dict[key_value]) == 0:
                    # 如果这个 supernode 没有邻居，就用它自己的特征
                    select_feature = whole_feature.iloc[key_value]
                else:
                    # 否则，将它与所有邻居的特征平均，作为 supernode 的表征
                    select_feature = whole_feature.iloc[node_dict[key_value] + [key_value]]
                    select_feature = select_feature.mean()
                # 填入 supernode_feature_strict
                if len(supernode_relate_value[2]) == 0:
                    temp_select = np.array(select_feature)
                    supernode_relate_value[2] = np.reshape(temp_select, (1,1794))
                else:
                    temp_select = np.array(select_feature)
                    supernode_relate_value[2] = np.concatenate((supernode_relate_value[2], np.reshape(temp_select, (1,1794))), axis=0)
                # 最终 supernode_relate_value[2] 是一个矩阵，shape 为 [N_supernode, 1794]
                pbar_node.update()

        # 构造超节点的空间坐标表
        coordinate_integrate = pd.DataFrame({'X':supernode_relate_value[0],'Y':supernode_relate_value[1]})
        # 计算所有 supernode 之间的欧氏距离
        coordinate_matrix1 = euclidean_distances(coordinate_integrate, coordinate_integrate)
        # 按空间阈值建立边连接，如果两个 supernode 的距离小于等于 spatial_threshold，就设为 1（连边）
        coordinate_matrix1 = np.where(coordinate_matrix1 > spatial_threshold , 0 , 1)     #空间距离

        fromlist = []
        tolist = []
        # 生成 PyTorch Geometric 所需的 edge_index
        with tqdm(total = len(coordinate_matrix1)) as pbar_pytorch_geom:
            for i in range(len(coordinate_matrix1)):
                temp = coordinate_matrix1[i,:]
                selectindex = np.where(temp > 0)[0].tolist()
                for index in selectindex:
                    fromlist.append(int(i))
                    tolist.append(int(index))
                pbar_pytorch_geom.update()
        # 构造 Data 对象
        # edge_index 是图的边，维度为 [2, num_edges]，表示有向边的连接关系
        edge_index = torch.tensor([fromlist, tolist], dtype=torch.long)
        # x 是所有 supernode 的特征矩阵，维度为 [N_supernode, 1794]
        x = torch.tensor(supernode_relate_value[2], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)

        node_dict = pd.DataFrame.from_dict(node_dict, orient='index')
        # 特征筛选聚合后的邻接表
        node_dict.to_csv(os.path.join(superpatch_dir, sample + '_' + str(threshold) + '.csv'))
        torch.save(data, os.path.join(superpatch_dir, sample+ '_' + str(threshold) + '_graph_torch.pt'))

def Parser_main():

    parser = argparse.ArgumentParser(description="TEA-graph superpatch generation")
    parser.add_argument("--graphdir",default=r"F:\graph",help="graph save dir",type=str)
    parser.add_argument("--imagedir",default=r"F:\selected_svs",help="svs file location",type=str)
    parser.add_argument("--weight_path",default=r"D:\PycharmProjects\zhongshan_hospital\Graph_neural_network\Superpatch_network_construction\EfficientNet\best.pth",help="pretrained weight path",type=str)
    #parser.add_argument("--weight_path",default=None,help="pretrained weight path",type=str)
    parser.add_argument("--imagesize", default = 256, help ="crop image size", type = int)
    parser.add_argument("--threshold", default = 0.75, help = "cosine similarity threshold", type = float)
    parser.add_argument("--spatial_threshold", default = 5.5, help = "spatial threshold", type = float)
    parser.add_argument("--gpu", default = '0' , help = "gpu device number", type = str)
    return parser.parse_args()
'''
class nisanf():
    def __init__(self, input_dim, out_dim):
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(self.input_dim, self.out_dim)
    
    def forward(x):
        out = self.linear(x)
        return out 

'''
def main():

    Argument = Parser_main()
    image_dir = Argument.imagedir
    save_dir = Argument.graphdir
    gpu = Argument.gpu
    files = os.listdir(image_dir)

    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)

    final_files = [os.path.join(image_dir, file) for file in files]
    final_files.sort(key=lambda f: os.stat(f).st_size, reverse=False)

    device = torch.device(int(gpu) if torch.cuda.is_available() else "cpu")
    model_ft = EfficientNet.from_pretrained('efficientnet-b4', num_classes = 2)
    if Argument.weight_path is not None:
        weight_path = Argument.weight_path
        load_weight = torch.load(weight_path, map_location = device)
        model_ft.load_state_dict(load_weight)
        print("预训练参数加载成功")

    model_ft = model_ft.to(device)
    model_ft.eval()

    with tqdm(total=len(final_files)) as pbar_tot:
        for image in final_files:
            # 检查是否为 .svs 文件
            if image.endswith('.svs'):
                # 获取不带扩展名的文件名
                filename, ext = os.path.splitext(image)
                # 如果文件名已经存在 .pt 的文件，则跳过
                filename = filename.split('.')[0]
                if os.path.exists(f"{filename}_0.75_graph_torch.pt"):
                    print(f"Skipping {image} because {filename}_0.75_graph_torch.pt already exists.")
                    continue
                # 你的处理代码
                print(f"Processing {image}")
                # try:
                supernode_generation(image, model_ft, device, Argument, save_dir)
                # except Exception as e:
                #     print(f"Error on {image}, skip and continue. Reason: {e}")
                #     pbar_tot.update()
                #     continue
                pbar_tot.update()

    # false_graph_filtering(4.3)

if __name__ == "__main__":
    # main()
    false_graph_filtering(4.3)