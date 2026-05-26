#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 09:04:37 2020

@author: taliq
"""

import os

import argparse
import torch
import torch_geometric.transforms as T
import pandas as pd
import numpy as np

from torch import optim
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.loader import DataListLoader

from torch_geometric.nn import DataParallel
from torch_geometric.utils import dropout_adj
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from utils.utils import build_independent_test_set
from utils.utils import coxph_loss
from utils.utils import accuracytest

from model_selection import model_selection

class CoxGraphDataset(Dataset):

    def __init__(self, filelist, survlist, stagelist, censorlist, Metadata, mode, model,
                 transform=None, pre_transform=None):
        super(CoxGraphDataset, self).__init__()
        self.filelist = filelist
        self.survlist = survlist
        self.stagelist = stagelist
        self.censorlist = censorlist
        self.Metadata = Metadata
        self.mode = mode
        self.model = model

    def processed_file_names(self):
        return self.filelist

    def len(self):
        return len(self.filelist)

    def get(self, idx):
        data_origin = torch.load(self.filelist[idx])
        transfer = T.ToSparseTensor()
        item = self.filelist[idx].split('/')[-1].split('.pt')[0].split('_')[0]
        mets_class = 0

        survival = self.survlist[idx]
        phase = self.censorlist[idx]
        stage = self.stagelist[idx]

        data_re = Data(x=data_origin.x[:, :1792], edge_index=data_origin.edge_index)
        data = transfer(data_re)
        data.survival = torch.tensor(survival)
        data.phase = torch.tensor(phase)
        data.mets_class = torch.tensor(mets_class)
        data.stage = torch.tensor(stage)
        data.item = item
        data.edge_attr = data_origin.edge_attr
        data.pos = data_origin.pos

        return data


def Analyze(Argument):

    batch_num = int(Argument.batch_size)
    # batch_num = Argument.batch_size
    device = torch.device(int(Argument.gpu))
    Metadata = pd.read_excel(r"E:\25-12-31-TCGA\metadata.xlsx")
    #Metadata = pd.read_excel("C:/Users/DerrickRose/Desktop/north_war.xlsx")

    #TrainRoot = TrainValid_path(Argument.DatasetType)
    #TrainRoot = 'E:/BaiduNetdiskDownload/pttest'
    TrainRoot = r"E:\25-12-31-TCGA\graph\superpatch"
    Trainlist = os.listdir(TrainRoot)
    Trainlist = [item for c, item in enumerate(Trainlist) if '0.75_graph_torch_4.3_artifact_sophis_final.pt' in item]
    #Trainlist = Trainlist[:100]
    Fi = Argument.FF_number

    # Test_set = train_test_split(Trainlist, Metadata, Argument.DatasetType, TrainRoot, Fi, Analyze_flag=True)
    Test_set = build_independent_test_set(Trainlist, Metadata, Argument.DatasetType, TrainRoot)
    TestDataset = CoxGraphDataset(filelist=Test_set[0], survlist=Test_set[1],
                                  stagelist=Test_set[3], censorlist=Test_set[2],
                                  Metadata=Metadata, mode=Argument.DatasetType,
                                  model=Argument.model)
    test_loader = DataListLoader(TestDataset, batch_size=batch_num, shuffle=True, num_workers=1, pin_memory=True,
                                 drop_last=False)

    #model = best_model
    #model = torch.load(Argument.load_state_dict, map_location="cpu")
    model = model_selection(Argument)

    model = DataParallel(model, device_ids=[0], output_device=0)
    model = model.to(device)



    #model.load_state_dict(torch.load("./results/TCGA/PatchGCN/2023-01-01_11_43_03/epoch-3,acc-0.083333,loss-0.904640.pt"))
    model.load_state_dict(torch.load(r"D:\PycharmProjects\zhongshan_hospital\Graph_neural_network\results\TCGA\MPGAT\2026-03-24_00_00_12\epoch-14,acc-0.644529,loss-0.797722.pt"), strict=False)# , strict=False

    # model = DataParallel(model, device_ids=[0, 1], output_device=0)
    model = model.to(device)
    Cox_loss = coxph_loss()
    Cox_loss = Cox_loss.to(device)

    EpochSurv = []
    EpochPhase = []
    EpochRisk = []
    EpochFeature = []
    EpochID = []
    EpochStage = []

    Epochloss = 0
    batchcounter = 1

    model.eval()
    count = 0
    idlist = np.empty([0, 1])
    slist = np.empty([0, 1])
    dlist = np.empty([0, 1])
    outlist = np.empty([0, 1])

    with tqdm(total=len(test_loader)) as pbar:
        with torch.set_grad_enabled(False):
            for c, d in enumerate(test_loader, 1):
                tempsurvival = torch.tensor([data.survival for data in d])
                survival = np.asarray([data.survival for data in d])
                survival = survival.reshape(len(survival), 1)
                tempphase = torch.tensor([data.phase for data in d])
                status = np.asarray([data.phase for data in d])
                status = status.reshape(len(status), 1)
                tempID = np.asarray([data.item for data in d])
                tempID = tempID.reshape(len(survival), 1)
                tempstage = torch.tensor([data.stage for data in d])
                tempmeta = torch.tensor([data.mets_class for data in d])
                idlist = np.concatenate((idlist, tempID), axis=0)
                slist = np.concatenate((slist, survival), axis=0)
                dlist = np.concatenate((dlist, status), axis=0)
                out = model(d)
                print("out:", out)

                out1 = out.cpu().numpy()
                print("out1:", out1)
                out1 = out1.reshape(-1, 1)
                outlist = np.concatenate((outlist, out1), axis=0)
                #out = F.normalize(out, p=2, dim=0)

                # final_updated_feature_list = updated_feature_list
                sort_idx = torch.argsort(tempsurvival, descending=True)

                risklist = out[sort_idx]
                tempsurvival = tempsurvival[sort_idx]
                tempphase = tempphase[sort_idx]
                for idx in sort_idx.cpu().detach().tolist():
                    EpochID.append(tempID[idx])
                tempstage = tempstage[sort_idx]
                tempmeta = tempmeta.to(out.device)

                risklist = risklist.to(out.device)
                tempsurvival = tempsurvival.to(out.device)
                tempphase = tempphase.to(out.device)
                # print("not utils_risklist:", risklist)
                # print("not utils_sur:", tempsurvival)
                # print("not utils_cens:", tempphase)

                if len(risklist) == 0:
                    temp = 0

                for riskval, survivalval, phaseval, stageval in zip(risklist, tempsurvival, tempphase, tempstage):
                    count = count + 1
                    EpochSurv.append(survivalval.cpu().detach().item())
                    EpochPhase.append(phaseval.cpu().detach().item())
                    if Argument.DatasetType == "BORAME_Meta":
                        EpochRisk.append(riskval[0].cpu().detach().item())
                    else:
                        EpochRisk.append(riskval.cpu().detach().item())
                    EpochStage.append(stageval.cpu().detach().item())

                    EpochPhase1 = np.asarray(EpochPhase).reshape(len(EpochPhase), 1)
                    EpochSurv1 = np.asarray(EpochSurv).reshape(len(EpochSurv), 1)
                    EpochRisk1 = np.asarray(EpochRisk).reshape(len(EpochRisk), 1)

                    idlist1 = np.empty([0, 1])
                    slist1 = np.empty([0, 1])
                    dlist1 = np.empty([0, 1])
                    outlist1 = np.empty([0, 1])
                    idlist1 = np.concatenate((idlist1, EpochID), axis=0)
                    slist1 = np.concatenate((slist1, EpochSurv1),axis=0)
                    dlist1 = np.concatenate((dlist1, EpochPhase1), axis=0)
                    outlist1 = np.concatenate((outlist1, EpochRisk1),axis=0)
                    # print(count)

                batchcounter += 1
                # if Argument.DatasetType == "BORAME_Meta":
                #     Batchacc = accuracytest(tempsurvival, risklist[:, 0], tempphase)
                # else:
                #     Batchacc = accuracytest(tempsurvival, risklist, tempphase)
                #
                # print("Batchacc:" + str(Batchacc))

                risklist = []
                tempsurvival = []
                tempphase = []
                tempstage = []

                pbar.update()
            pbar.close()
    # print("EpochRisk:", EpochRisk)
    # print("EpochSurv:", EpochSurv)
    # print("EpochCens:", EpochPhase)
    a = np.concatenate((idlist, slist, dlist, outlist), axis=1)
    # print(a.shape)
    b = pd.DataFrame(a)
    b.to_excel(r'D:\PycharmProjects\zhongshan_hospital\Graph_neural_network\results\TCGA\MPGAT\2026-03-24_00_00_12\tcga5.xlsx')

    # c = np.concatenate((idlist1, slist1, dlist1, outlist1), axis=1)
    # d = pd.DataFrame(c)
    # d.to_excel('E:\TCGA_LIHC\output2.xlsx')

    Epochacc = accuracytest(torch.tensor(EpochSurv), torch.tensor(EpochRisk), torch.tensor(EpochPhase))

    print(" acc:" + str(Epochacc))
    #statistical_vis(Figure_dir, (EpochSurv, EpochRisk, EpochStage, EpochPhase, EpochID), epoch)



    return 0

def Parser_main():
    parser = argparse.ArgumentParser(description="Deep cox analysis model")
    parser.add_argument("--DatasetType", default="TCGA", help="TCGA_BRCA or BORAME or BORAME_Meta or BORAME_Prog",
                        type=str)
    parser.add_argument("--learning_rate", default=0.0001, help="Learning rate", type=float)
    parser.add_argument("--weight_decay", default=0.00005, help="Weight decay rate", type=float)
    parser.add_argument("--clip_grad_norm_value", default=2.0, help="Gradient clipping value", type=float)
    parser.add_argument("--batch_size", default=4, help="batch size", type=int)
    parser.add_argument("--num_epochs", default=50, help="2", type=int)
    parser.add_argument("--dropedge_rate", default=0.25, help="Dropedge rate for GAT", type=float)
    parser.add_argument("--dropout_rate", default=0.25, help="Dropout rate for MLP", type=float)
    parser.add_argument("--graph_dropout_rate", default=0.25, help="Node/Edge feature dropout rate", type=float)
    parser.add_argument("--initial_dim", default=100, help="Initial dimension for the GAT", type=int)
    parser.add_argument("--attention_head_num", default=2, help="Number of attention heads for GAT", type=int)
    parser.add_argument("--number_of_layers", default=5, help="Whole number of layer of GAT", type=int)
    parser.add_argument("--FF_number", default=1, help="Selecting set for the five fold cross validation", type=int)
    parser.add_argument("--model", default="MPGAT", help="GAT_custom/DeepGraphConv/PatchGCN/GIN/MIL/MIL-attention", type=str)
    parser.add_argument("--gpu", default=0, help="Target gpu for calculating loss value", type=int)
    parser.add_argument("--norm_type", default="layer", help="BatchNorm=batch/LayerNorm=layer", type=str)
    parser.add_argument("--MLP_layernum", default=3, help="Number of layers for pre/pose-MLP", type=int)
    parser.add_argument("--with_distance", default="Y", help="Y/N; Including positional information as edge feature", type=str)
    parser.add_argument("--simple_distance", default="N", help="Y/N; Whether multiplying or embedding positional information", type=str)
    parser.add_argument("--loss_type", default="PRELU", help="RELU/Leaky/PRELU", type=str)
    parser.add_argument("--residual_connection", default="Y", help="Y/N", type=str)


    return parser.parse_args()

def main():
    Argument = Parser_main()

    Analyze(Argument)

if __name__ == "__main__":
    main()