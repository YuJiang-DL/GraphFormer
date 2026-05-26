# -*- coding: utf-8 -*-

import copy
import torch
import torch_geometric.transforms as T
import pandas as pd
import numpy as np

from torch import optim
from torch_geometric.transforms import Polar
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import DataParallel
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, OneCycleLR
#OneCycleLR

from tqdm import tqdm

from model_selection import model_selection
from utils.utils import train_valid_split_only, build_independent_test_set
from utils.utils import makecheckpoint_dir_graph as mcd
from utils.utils import TrainValid_path
from utils.utils import non_decay_filter
from utils.utils import coxph_loss
from utils.utils import cox_sort
from utils.utils import accuracytest

from torch.utils.data.sampler import Sampler

class Sampler_custom(Sampler):

    def __init__(self, event_list, censor_list, batch_size):
        self.event_list = event_list
        self.censor_list = censor_list
        self.batch_size = batch_size

    def __iter__(self):

        train_batch_sampler = []
        #print(len(self.event_list))
        Event_idx = copy.deepcopy(self.event_list)
        Censored_idx = copy.deepcopy(self.censor_list)
        # np.random.shuffle(Event_idx)
        # np.random.shuffle(Censored_idx)
        #print("Event_idx:",Event_idx)
        Int_event_batch_num = Event_idx.shape[0] // 2
        Int_event_batch_num = Int_event_batch_num * 2
        Event_idx_batch_select = np.random.choice(Event_idx.shape[0], Int_event_batch_num, replace=False)
        Event_idx = Event_idx[Event_idx_batch_select]

        Int_censor_batch_num = Censored_idx.shape[0] // (self.batch_size - 2)
        #Int_censor_batch_num = Censored_idx.shape[0] // (self.batch_size - 0)
        Int_censor_batch_num = Int_censor_batch_num * (self.batch_size - 2)
        #Int_censor_batch_num = Int_censor_batch_num * (self.batch_size - 0)
        Censored_idx_batch_select = np.random.choice(Censored_idx.shape[0], Int_censor_batch_num, replace=False)
        Censored_idx = Censored_idx[Censored_idx_batch_select]

        #print("Event_idx:",Event_idx)
        #print("len(Event_idx) // 2:", len(Event_idx) // 2)
        Event_idx_selected = np.random.choice(Event_idx, size=(len(Event_idx) // 2, 2), replace=False)
        Censored_idx_selected = np.random.choice(Censored_idx, size=(
            (Censored_idx.shape[0] // (self.batch_size - 2)), (self.batch_size - 2)), replace=False)  # -2 -> - 0
        #print("Event_idx_selected:", Event_idx_selected)
        #print("Censored_idx_selected:", Censored_idx_selected)
###########################################################################################

        if Event_idx_selected.shape[0] > Censored_idx_selected.shape[0]:
            Event_idx_selected = Event_idx_selected[:Censored_idx_selected.shape[0],:]  #  [:0,:]
        else:
            Censored_idx_selected = Censored_idx_selected[:Event_idx_selected.shape[0],:]
        #print("Event_idx_selected 2:", Event_idx_selected)

        for c in range(Event_idx_selected.shape[0]):
            train_batch_sampler.append(
                Event_idx_selected[c, :].flatten().tolist() + Censored_idx_selected[c, :].flatten().tolist())

        #print("train_batch_sampler",train_batch_sampler)
        return iter(train_batch_sampler)

    def __len__(self):
        return len(self.event_list) // 2
        #return len(self.event_list) // 8

class CoxGraphDataset(Dataset):

    def __init__(self, filelist, survlist, stagelist, censorlist, Metadata, mode, model, transform=None, pre_transform=None):
        super(CoxGraphDataset, self).__init__()
        self.filelist = filelist
        self.survlist = survlist
        self.stagelist = stagelist
        self.censorlist = censorlist
        self.Metadata = Metadata
        self.mode = mode
        self.model = model
        self.polar_transform = Polar()

    def processed_file_names(self):
        return self.filelist

    def len(self):
        return len(self.filelist)

    def get(self, idx):
        data_origin = torch.load(self.filelist[idx])
        #print("self.filelist0", self.filelist[idx])
        transfer = T.ToSparseTensor()
        item = self.filelist[idx].split('/')[-1].split('.pt')[0].split('_')[0]
        mets_class = 0

        survival = self.survlist[idx]
        phase = self.censorlist[idx]
        stage = self.stagelist[idx]

        data_re = Data(x=data_origin.x[:,:1792], edge_index=data_origin.edge_index)

        mock_data = Data(x=data_origin.x[:,:1792], edge_index=data_origin.edge_index, pos=data_origin.pos)

        data_re.pos = data_origin.pos
        data_re_polar = self.polar_transform(mock_data)
        polar_edge_attr = data_re_polar.edge_attr

        if (data_re.edge_index.shape[1] != data_origin.edge_attr.shape[0]):
            print('error!')
            #print(self.filelist[idx].split('/')[-1])
        else:
            data = transfer(data_re)
            data.survival = torch.tensor(survival)
            data.phase = torch.tensor(phase)
            data.mets_class = torch.tensor(mets_class)
            data.stage = torch.tensor(stage)
            data.item = item
            data.edge_attr = polar_edge_attr
            data.pos = data_origin.pos

        return data

def Train(Argument):
    import os
    import numpy as np
    import pandas as pd
    import torch
    from torch import optim
    from torch.optim.lr_scheduler import OneCycleLR
    from torch_geometric.loader import DataListLoader
    from torch_geometric.nn import DataParallel

    checkpoint_dir, Figure_dir = mcd(Argument)

    batch_num = int(Argument.batch_size)
    gpu_id = int(getattr(Argument, "gpu", 0))
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    Metadata = pd.read_excel(r"E:\metedata.xlsx")
    # Metadata1 = pd.read_excel(r"F:\metedata.xlsx")
    Metadata2 = pd.read_excel(r"F:\PD-1 RCT结果-ITT 修正.xlsx")
    TrainRoot = TrainValid_path(Argument.DatasetType)
    # Test1Root = r"F:\graph\superpatch"
    TestRoot = r"F:\superpatch"
    Trainlist = os.listdir(TrainRoot)
    Trainlist = [x for x in Trainlist if '0.75_graph_torch_4.3_artifact_sophis_final.pt' in x]
    TestList = os.listdir(TestRoot)
    TestList = [item for item in TestList if '0.75_graph_torch_4.3_artifact_sophis_final.pt' in item]
    Fi = Argument.FF_number
    # TrainFF_set, ValidFF_set, Test_set = train_test_split(
    #     Trainlist, Metadata, Argument.DatasetType, TrainRoot, Fi
    # )
    TrainFF_set, ValidFF_set = train_valid_split_only(Trainlist, Metadata, Argument.DatasetType, TrainRoot, Fi)
    Test_set = build_independent_test_set(TestList, Metadata2, Argument.DatasetType, TestRoot)

    TrainDataset = CoxGraphDataset(filelist=TrainFF_set[0], survlist=TrainFF_set[1],
                                   stagelist=TrainFF_set[3], censorlist=TrainFF_set[2],
                                   Metadata=Metadata, mode=Argument.DatasetType,
                                   model=Argument.model)
    #
    ValidDataset = CoxGraphDataset(filelist=ValidFF_set[0], survlist=ValidFF_set[1],
                                   stagelist=ValidFF_set[3], censorlist=ValidFF_set[2],
                                   Metadata=Metadata, mode=Argument.DatasetType,
                                   model=Argument.model)

    TestDataset = CoxGraphDataset(filelist=Test_set[0], survlist=Test_set[1],
                                  stagelist=Test_set[3], censorlist=Test_set[2],
                                  Metadata=Metadata2, mode=Argument.DatasetType,
                                  model=Argument.model)
    # TrainDataset = TestDataset
    # ValidDataset = TestDataset
    print("len(TrainDataset):", len(TrainDataset))
    print("len(ValidDataset):", len(ValidDataset))
    print("len(TestDataset):", len(TestDataset))
    # --- sampler（保持你原来的逻辑）---
    Event_idx = np.where(np.array(TrainFF_set[2]) == 1)[0]
    Censored_idx = np.where(np.array(TrainFF_set[2]) == 0)[0]
    train_batch_sampler = Sampler_custom(Event_idx, Censored_idx, batch_num)

    torch.manual_seed(12345)

    # ✅保持你原来的 DataListLoader（返回 list[Data]）
    test_loader = DataListLoader(TestDataset, batch_size=batch_num, shuffle=False,
                                 num_workers=1, pin_memory=True, drop_last=False)

    train_loader = DataListLoader(TrainDataset, batch_sampler=train_batch_sampler, shuffle=False,
                                  num_workers=1, pin_memory=True)

    val_loader = DataListLoader(ValidDataset, batch_size=batch_num, shuffle=False,
                                num_workers=1, pin_memory=True, drop_last=False)

    loader = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    # --- model（按你原来的方式：必须 DataParallel 包起来）---
    model = model_selection(Argument)
    model_parameter_groups = non_decay_filter(model)

    # ✅关键：用 PyG 的 DataParallel，且 device_ids 要跟 Argument.gpu 一致
    model = DataParallel(model, device_ids=[gpu_id], output_device=gpu_id)
    model = model.to(device)
# D:\PycharmProjects\zhongshan_hospital\Graph_neural_network\results\TCGA\MPGAT\B6C1NA\epoch-37,acc-0.820115,loss-0.994918.pt
    Cox_loss = coxph_loss().to(device)   # 你原来用的

    optimizer_ft = optim.AdamW(model_parameter_groups,
                               lr=Argument.learning_rate,
                               weight_decay=Argument.weight_decay)

    if len(train_loader) <= 0:
        raise ValueError(f"train_loader length is {len(train_loader)}. Check sampler / dataset.")

    scheduler = OneCycleLR(optimizer_ft, max_lr=Argument.learning_rate,
                           steps_per_epoch=len(train_loader),
                           epochs=int(Argument.num_epochs))

    bestloss = float("inf")
    bestacc = 0.0
    bestepoch = 0

    checkpointinfo = 'epoch-{},acc-{:.6f},loss-{:.6f}.pt'

    for epoch in range(int(Argument.num_epochs)):
        for mode in ['train', 'val', 'test']:
            is_train = (mode == 'train')
            model.train() if is_train else model.eval()

            Epochloss = 0.0
            batchcounter = 0
            pass_count = 0

            EpochSurv, EpochPhase, EpochRisk, EpochStage, EpochID = [], [], [], [], []
            # 专门给 test 导出 csv 用
            TestRowList = []
            # ✅关键：把整个 loop 放到 set_grad_enabled 里
            with torch.set_grad_enabled(is_train):
                for step, d in enumerate(loader[mode], 1):
                    # d 是 list[Data] —— 这是你以前能跑通的输入形式
                    if is_train:
                        optimizer_ft.zero_grad(set_to_none=True)

                    tempsurvival = torch.tensor([data.survival for data in d])
                    tempphase = torch.tensor([data.phase for data in d])
                    tempID = np.asarray([data.item for data in d])
                    tempstage = torch.tensor([data.stage for data in d])
                    tempmeta = torch.tensor([data.mets_class for data in d])

                    out = model(d)
                    # 只在 test 模式下，逐 batch 收集所有样本输出
                    if mode == 'test':
                        out_cpu = out.detach().cpu().view(-1).numpy()
                        survival_cpu = np.array([data.survival.cpu().detach().item() for data in d]).reshape(-1)
                        phase_cpu = np.array([data.phase.cpu().detach().item() for data in d]).reshape(-1)
                        stage_cpu = np.array([data.stage.cpu().detach().item() for data in d]).reshape(-1)
                        item_cpu = [str(data.item) for data in d]

                        for sample_id, surv_v, phase_v, stage_v, risk_v in zip(
                                item_cpu, survival_cpu, phase_cpu, stage_cpu, out_cpu):
                            TestRowList.append({
                                "item": sample_id,
                                "survival": surv_v,
                                "phase": phase_v,
                                "stage": stage_v,
                                "risk_score": float(risk_v)
                            })
                    risklist, tempsurvival, tempphase, tempmeta, EpochSurv, EpochPhase, EpochRisk, EpochStage = \
                        cox_sort(out, tempsurvival, tempphase, tempmeta, tempstage, tempID,
                                 EpochSurv, EpochPhase, EpochRisk, EpochStage, EpochID)

                    # ✅无事件 batch：直接跳过，不参与统计/反传
                    if torch.sum(tempphase).item() < 1:
                        pass_count += 1
                        continue

                    loss = Cox_loss(risklist, tempsurvival, tempphase)

                    if is_train:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model_parameter_groups[0]['params'],
                                                       max_norm=Argument.clip_grad_norm_value,
                                                       error_if_nonfinite=True)
                        torch.nn.utils.clip_grad_norm_(model_parameter_groups[1]['params'],
                                                       max_norm=Argument.clip_grad_norm_value,
                                                       error_if_nonfinite=True)
                        optimizer_ft.step()
                        scheduler.step()

                    Epochloss += float(loss.item())
                    batchcounter += 1

            Epochacc = accuracytest(torch.tensor(EpochSurv), torch.tensor(EpochRisk), torch.tensor(EpochPhase))
            Epochloss_avg = Epochloss / max(batchcounter, 1)

            print(f"epoch:{epoch}  mode:{mode}  loss:{Epochloss_avg:.6f}  acc:{Epochacc:.6f}  pass count:{pass_count}")

            if mode == 'test':
                test_df = pd.DataFrame(TestRowList)

                metadata_id_col = "ID"  # 改成你的 Metadata2 真实主键列名

                if metadata_id_col in Metadata2.columns:
                    Metadata2_copy = Metadata2.copy()
                    Metadata2_copy[metadata_id_col] = Metadata2_copy[metadata_id_col].astype(str)
                    test_df["item"] = test_df["item"].astype(str)

                    test_merge_df = test_df.merge(
                        Metadata2_copy,
                        left_on="item",
                        right_on=metadata_id_col,
                        how="left"
                    )
                else:
                    print(f"[Warn] Metadata2 中不存在列: {metadata_id_col}，仅保存 test 输出。")
                    test_merge_df = test_df
                # 只按 best acc 保存
                if epoch == 0 or Epochacc > bestacc:
                    model_save_path = os.path.join(
                        checkpoint_dir,
                        checkpointinfo.format(epoch, Epochacc, Epochloss_avg)
                    )
                    torch.save(model.state_dict(), model_save_path)

                    csv_save_path = os.path.join(checkpoint_dir, "best_test_predictions.csv")
                    test_merge_df.to_csv(csv_save_path, index=False, encoding="utf-8-sig")

                    bestacc = Epochacc
                    bestloss = Epochloss_avg
                    bestepoch = epoch

                    print(f"best model saved to: {model_save_path}")
                    print(f"best test csv saved to: {csv_save_path}")

                # 只更新 bestloss，但不保存 csv
                if Epochloss_avg < bestloss:
                    bestloss = Epochloss_avg

    return model, checkpoint_dir, Figure_dir, bestepoch