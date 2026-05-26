# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from models.PatchGCN import PatchGCN_module as gcn
from torch_geometric.transforms import ToSparseTensor
from torch.nn import LayerNorm
from torch_geometric.nn import global_mean_pool, BatchNorm
from models.Modified_GAT import GATConv as GATConv
from torch_geometric.nn import GraphSizeNorm
from torch_sparse import to_torch_sparse, SparseTensor
from models.model_utils import weight_init
from models.model_utils import decide_loss_type
#from ceshi2 import MyViT
from models.pre_layer import preprocess
from models.post_layer import postprocess


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1792, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class GAT_module(torch.nn.Module):

    def __init__(self, input_dim, output_dim, head_num, dropedge_rate, graph_dropout_rate, loss_type, with_edge, simple_distance, norm_type):
        """
        :param input_dim: Input dimension for GAT
        :param output_dim: Output dimension for GAT
        :param head_num: number of heads for GAT
        :param dropedge_rate: Attention-level dropout rate
        :param graph_dropout_rate: Node/Edge feature drop rate
        :param loss_type: Choose the loss type
        :param with_edge: Include the edge feature or not
        :param simple_distance: Simple multiplication of edge feature or not
        :param norm_type: Normalization method
        """

        super(GAT_module, self).__init__()
        self.conv = GATConv([input_dim, input_dim], output_dim, heads=head_num, dropout=dropedge_rate, with_edge=with_edge, simple_distance=simple_distance)
        self.norm_type = norm_type
        if norm_type == "layer":
            self.bn = LayerNorm(output_dim * int(self.conv.heads))
            self.gbn = None
        else:
            self.bn = BatchNorm(output_dim * int(self.conv.heads))
            self.gbn = GraphSizeNorm()
        self.prelu = decide_loss_type(loss_type, output_dim * int(self.conv.heads))
        self.dropout_rate = graph_dropout_rate
        self.with_edge = with_edge

    def reset_parameters(self):

        self.conv.reset_parameters()
        self.bn.reset_parameters()

    def forward(self, x, edge_attr, edge_index, batch):

        if self.training:
            drop_node_mask = x.new_full((x.size(1),), 1 - self.dropout_rate, dtype=torch.float)
            drop_node_mask = torch.bernoulli(drop_node_mask)
            drop_node_mask = torch.reshape(drop_node_mask, (1, drop_node_mask.shape[0]))
            drop_node_feature = x * drop_node_mask

            drop_edge_mask = edge_attr.new_full((edge_attr.size(1),), 1 - self.dropout_rate, dtype=torch.float)
            drop_edge_mask = torch.bernoulli(drop_edge_mask)
            drop_edge_mask = torch.reshape(drop_edge_mask, (1, drop_edge_mask.shape[0]))
            drop_edge_attr = edge_attr * drop_edge_mask
        else:
            drop_node_feature = x
            drop_edge_attr = edge_attr

        if self.with_edge == "Y":
            x_before, attention_value = self.conv((drop_node_feature, drop_node_feature), edge_index,
                                   edge_attr=drop_edge_attr, return_attention_weights=True)
        else:
            x_before, attention_value = self.conv((drop_node_feature, drop_node_feature), edge_index,
                                   edge_attr=None, return_attention_weights=True)
        out_x_temp = 0
        if self.norm_type == "layer":
            for c, item in enumerate(torch.unique(batch)):
                temp = self.bn(x_before[batch == item])
                if c == 0:
                    out_x_temp = temp
                else:
                    out_x_temp = torch.cat((out_x_temp, temp), 0)
        else:
            temp = self.gbn(self.bn(x_before), batch)
            out_x_temp = temp

        x_after = self.prelu(out_x_temp)

        return x_after, attention_value

class GraphFormer(torch.nn.Module):

    def __init__(self, dropout_rate, dropedge_rate, Argument):
        super(GraphFormer, self).__init__()
        torch.manual_seed(12345)
        self.Argument = Argument

        dim = Argument.initial_dim
        self.dropout_rate = dropout_rate
        self.dropedge_rate = dropedge_rate
        self.heads_num = Argument.attention_head_num
        self.include_edge_feature = Argument.with_distance
        self.layer_num = Argument.number_of_layers
        self.graph_dropout_rate = Argument.graph_dropout_rate
        self.residual = Argument.residual_connection
        self.norm_type = Argument.norm_type
        self.output_act = nn.Tanh()

        self.gcn = nn.ModuleList([gcn(Argument.initial_dim * Argument.attention_head_num, 1, Argument.dropout_rate) for _ in  range(int(Argument.number_of_layers))])
        # self.ls1=XLSTM_Transformer(200, 100, 2,  200, output_dim=200)
        postNum = 0
        self.preprocess = preprocess(Argument)
        self.conv_list = nn.ModuleList([GAT_module(dim * self.heads_num, dim, self.heads_num, self.dropedge_rate,
                                                   self.graph_dropout_rate, Argument.loss_type,
                                                   with_edge=Argument.with_distance,
                                                   simple_distance=Argument.simple_distance,
                                                   norm_type=Argument.norm_type) for _ in
                                        range(int(Argument.number_of_layers))])
        # postNum += int(self.heads_num) * len(self.conv_list)
        # self.l1= nn.Linear(200, 200)
        self.postprocess = postprocess(dim * self.heads_num, self.layer_num, dim * self.heads_num,
                                       (Argument.MLP_layernum - 1), dropout_rate)
        # layer5 300, layer3 200
        self.encoder_layer0 = torch.nn.TransformerEncoderLayer(
            d_model=300,
            nhead=2,
            dropout=0.1,
            dim_feedforward=6 * 50,
        )
        self.encoder0 = torch.nn.TransformerEncoder(self.encoder_layer0, num_layers=8)

        # self.decoder_layer0 = torch.nn.TransformerDecoderLayer(
        #     d_model=300,
        #     nhead=2,
        #     dropout=0.1,
        #     dim_feedforward=6 * 50,
        # )
        # self.decoder0 = torch.nn.TransformerDecoder(self.decoder_layer0, num_layers=8)
        self.risk_prediction_layer = nn.Sequential(
            nn.Linear(self.postprocess.postlayernum[-1], 1))

    def reset_parameters(self):

        self.preprocess.reset_parameters()
        for i in range(int(self.Argument.number_of_layers)):
            self.conv_list[i].reset_parameters()
        self.postprocess.reset_parameters()
        self.lstm1.reset_parameters()
        self.lin1.reset_parameters()
        self.lstm2.reset_parameters()
        self.risk_prediction_layer.apply(weight_init)

    def forward(self, data, edge_mask=None, Interpretation_mode=False):
        preprocessed_input, preprocess_edge_attr = self.preprocess(data, edge_mask)
        batch = data.batch
        # preprocessed_input = self.ls1(preprocessed_input)
        x0_glob = global_mean_pool(preprocessed_input, batch)

        x_concat = x0_glob

        x_out = preprocessed_input
        final_x = x_out
        count = 0
        attention_list = []

        L = 1


        for i in range(int(self.layer_num)):
            select_idx = int(i)
            x_out_gcn1 = self.gcn[select_idx](x_out, data.adj_t)
            x_adj=torch.mm(x_out_gcn1,x_out_gcn1.T)
            data.adj_t=(x_adj*data.adj_t.to_dense()).to_sparse()
            data.adj_t = SparseTensor(row=data.adj_t.indices()[0], col=data.adj_t.indices()[1],sparse_sizes=(data.adj_t.size()[0], data.adj_t.size()[0]))
            x_temp_out, attention_value = \
                self.conv_list[select_idx](x_out, preprocess_edge_attr, data.adj_t, batch)
            _, _, attention_value = attention_value.coo()
            if len(attention_list) == 0:
                attention_list = torch.reshape(attention_value, (1, attention_value.shape[0], attention_value.shape[1]))
            else:
                attention_list = torch.cat((attention_list, torch.reshape(attention_value, (
                    1, attention_value.shape[0], attention_value.shape[1]))), 0)
            #print('x_temp',x_temp_out.shape)
            if self.residual == "Y":

                x_out = x_temp_out + x_out_gcn1

            else:
                x_out = x_temp_out
            # x_glob = global_mean_pool(x_temp_out, batch)
            x_glob = global_mean_pool(x_out, batch)
            x_concat = torch.cat((x_concat, x_glob), 1)
            #print('x_glob',x_glob.shape)


            final_x = x_out
            count = count + 1
        # print('x',x_concat.shape)
        postprocessed_output = self.postprocess(x_concat, data.batch)
        # print(postprocessed_output.shape) #[6,300]
        postprocessed_output01 = self.encoder0(postprocessed_output)
        # postprocessed_output = self.decoder0(postprocessed_output, postprocessed_output01)
        risk = self.risk_prediction_layer(postprocessed_output01)
        # risk = 0.1 * self.output_act(risk)
        if Interpretation_mode:
            return risk, final_x, attention_list
        else:
            return risk
