
from models.GAT import GAT
from models.MLP import MLP
from models.GAT_custom import GAT as GAT_custom
from models.graphformer import GraphFormer as net
from models.MLP_attention import MLP_attention as AttnMLP
from models.DeepGraphConv import DeepGraphConv_Surv as DeepGraphConv
from models.PatchGCN import PatchGCN
from models.TransGCN import PatchGCN as gcn

def model_selection(Argument):

    if Argument.model == "GAT_TCGA_STAD":
        model = GAT(Argument.dropout_rate, Argument.dropedge_rate, Argument)
    elif Argument.model == "GAT_custom":
        model = GAT_custom(Argument.dropout_rate, Argument.dropedge_rate, Argument)
    elif Argument.model == "MLP_TCGA_STAD":
        model = MLP(Argument.dropout_rate, Argument.dropedge_rate, Argument)
    elif Argument.model == "AttMLP_TCGA_STAD":
        model = AttnMLP(Argument.dropout_rate, Argument.dropedge_rate, Argument)
    elif Argument.model == "PatchGCN":
        model = PatchGCN(Argument.dropout_rate, Argument.dropedge_rate, Argument)
    elif Argument.model == "DeepGraphConv_TCGA_STAD":
        model = DeepGraphConv(Argument.dropout_rate, Argument.dropedge_rate, Argument)
    elif Argument.model == "MPGAT":
        model = net(Argument.dropout_rate, Argument.dropedge_rate, Argument)
    elif Argument.model == "TransGCN":
        model = gcn(Argument.dropout_rate, Argument.dropedge_rate, Argument)
    else:
        print("Enter the valid model type")
        model = None

    return model