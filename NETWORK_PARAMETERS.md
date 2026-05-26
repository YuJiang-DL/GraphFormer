# Network Parameters Table

## GraphFormer Model Parameters

| Parameter | Description | Default Value / Source |
|-----------|-------------|------------------------|
| `initial_dim` | Initial dimension of node features | From `Argument.initial_dim` |
| `attention_head_num` | Number of attention heads in GAT layers | From `Argument.attention_head_num` |
| `with_distance` | Whether to include edge features (distance) | From `Argument.with_distance` |
| `number_of_layers` | Number of GAT/GCN layers | From `Argument.number_of_layers` |
| `graph_dropout_rate` | Dropout rate for node/edge features | From `Argument.graph_dropout_rate` |
| `residual_connection` | Whether to use residual connection in GAT layers | From `Argument.residual_connection` |
| `norm_type` | Type of normalization ("layer" or batch) | From `Argument.norm_type` |
| `loss_type` | Type of loss function | From `Argument.loss_type` |
| `simple_distance` | Whether to use simple multiplication for edge features | From `Argument.simple_distance` |
| `MLP_layernum` | Number of layers in the MLP postprocessor | From `Argument.MLP_layernum` |
| `dropout_rate` | Dropout rate in GCN layers and elsewhere | From `Argument.dropout_rate` (via `dropout_rate` parameter) |
| `dropedge_rate` | Dropout rate for edges in GAT layers | From `Argument.dropedge_rate` (via `dropedge_rate` parameter) |

## Fixed Architectural Parameters

| Component | Parameter | Value |
|-----------|-----------|-------|
| Transformer Encoder Layer | `d_model` | 300 |
| Transformer Encoder Layer | `nhead` (number of heads) | 2 |
| Transformer Encoder Layer | `dropout` | 0.1 |
| Transformer Encoder Layer | `dim_feedforward` | 300 (6 * 50) |
| Transformer Encoder | `num_layers` | 8 |
| GCN Layers (in ModuleList) | Output dimension | 1 |
| GCN Layers (in ModuleList) | Input dimension | `Argument.initial_dim * Argument.attention_head_num` |
| Risk Prediction Layer | Input dimension | `self.postprocess.postlayernum[-1]` |
| Risk Prediction Layer | Output dimension | 1 |

## Notes
- The `Argument` object is passed to the GraphFormer constructor and contains the main configurable hyperparameters.
- The Transformer encoder has fixed architecture parameters as defined in the code (not configurable via Argument).
- The postprocessor (`postprocess`) architecture depends on `Argument.MLP_layernum` and the hidden dimension.
- Actual parameter counts would depend on the specific values of the Argument fields and input data dimensions.
