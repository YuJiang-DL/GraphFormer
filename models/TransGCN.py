import torch
import torch.nn as nn

from torch.nn import ReLU, LayerNorm
from torch_geometric.nn import GENConv, DeepGCNLayer
from torch_geometric.utils import softmax
from torch_scatter import scatter_add

from models.model_utils import weight_init

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


class SimTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim,  nhead, d_model, output_dim=1):
        super(SimTransformer, self).__init__()
        self.transformer_encoder = TransformerEncoderLayer(d_model, nhead)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        transformed_out = self.transformer_encoder(x.unsqueeze(0)).squeeze(0)
        output = self.fc(transformed_out)
        return output

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)
        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def reset_parameters(self):

        self.attention_a.apply(weight_init)
        self.attention_b.apply(weight_init)
        self.attention_c.apply(weight_init)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x

class PatchGCN_module(torch.nn.Module):

    def __init__(self, hidden_dim, i, dropout_rate):

        super(PatchGCN_module, self).__init__()
        self.conv = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                       t=1.0, learn_t=True, num_layers=2, norm='layer')
        self.norm = LayerNorm(hidden_dim, elementwise_affine=True)
        self.act = ReLU(inplace=True)
        #self.layer = DeepGCNLayer(self.conv, self.norm, self.act, block='res', dropout=0.1, ckpt_grad=i % 3)
        self.layer = DeepGCNLayer(self.conv, self.norm, self.act, block='res', dropout=0.1)
        self.dropout_rate = dropout_rate

    def reset_parameters(self):

        self.layer.reset_parameters()

    def forward(self, x, edge_index):

        drop_node_mask = x.new_full((x.size(1),), 1 - self.dropout_rate, dtype=torch.float)
        drop_node_mask = torch.bernoulli(drop_node_mask)
        drop_node_mask = torch.reshape(drop_node_mask, (1, drop_node_mask.shape[0]))
        drop_node_feature = x * drop_node_mask

        x_after = self.layer(drop_node_feature, edge_index)

        return x_after

class PatchGCN(torch.nn.Module):

    def __init__(self, dropout_rate, dropedge_rate, Argument):
        super(PatchGCN, self).__init__()

        hidden_dim = Argument.initial_dim * Argument.attention_head_num
        self.num_layers = Argument.number_of_layers
        dropout = Argument.dropout_rate

        self.fc = nn.Sequential(*[nn.Linear(1792, hidden_dim), nn.ReLU(), nn.Dropout(0.25)])

        self.total_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.total_layers.append(PatchGCN_module(hidden_dim, i, dropout))

        self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim * 6, hidden_dim * 6), nn.ReLU(), nn.Dropout(0.25)])

        self.path_attention_head = Attn_Net_Gated(L=hidden_dim * 6, D=hidden_dim * 6, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim * 6, hidden_dim * 6), nn.ReLU(), nn.Dropout(dropout)])
        self.trans = SimTransformer(200, 100, 2, 200, output_dim=200)
        self.encoder_layer0 = torch.nn.TransformerEncoderLayer(
            d_model=1200,
            nhead=2,
            dropout=0.1,
            dim_feedforward=6 * 200,
        )
        self.encoder0 = torch.nn.TransformerEncoder(self.encoder_layer0, num_layers=8)

        self.decoder_layer0 = torch.nn.TransformerDecoderLayer(
            d_model=1200,
            nhead=2,
            dropout=0.1,
            dim_feedforward=6 * 200,
        )
        self.decoder0 = torch.nn.TransformerDecoder(self.decoder_layer0, num_layers=8)
        self.risk_prediction_layer = nn.Linear(hidden_dim * 6, 1)

    def reset_parameters(self):

        self.fc.apply(weight_init)
        for i in range(len(self.total_layers)):
            self.total_layers[i].reset_parameters()
        self.path_phi.apply(weight_init)
        self.path_rho.apply(weight_init)
        self.risk_prediction_layer.apply(weight_init)
        self.path_attention_head.reset_parameters()

    def forward(self, data):

        x = self.fc(data.x)
        # x=self.trans(x)
        x_ = x

        edge_index = data.adj_t
        edge_attr = None
        batch = data.batch

        x = self.total_layers[0].conv(x_, edge_index, edge_attr)
        x_ = torch.cat([x_, x], axis=1)
        for layer in self.total_layers[1:]:
            x = layer(x, edge_index)
            x_ = torch.cat([x_, x], axis=1)

        h_path = x_
        h_path = self.path_phi(h_path)
        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = scatter_add(torch.mul(h_path.permute(1,0), softmax(A_path.flatten(), batch)).permute(1,0), batch, dim=0)
        h = self.path_rho(h_path).squeeze()
        if h.dim() == 1:  # 检查 h 是否为 1D
            h = h.unsqueeze(0)  # 将其变为 2D，形状从 (N,) 变为 (1, N)

        postprocessed_output01 = self.encoder0(h)
        postprocessed_output = self.decoder0(h, postprocessed_output01)
        h = self.risk_prediction_layer(postprocessed_output).flatten()

        return h