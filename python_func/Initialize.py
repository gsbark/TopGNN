import torch 
import torch_geometric as pyg
from torch_geometric.nn import EdgeConv
import math
import numpy as np 

class MLP(torch.nn.Module):
    """Multi-Layer perceptron"""
    def __init__(self, input_size, hidden_size, output_size, layers, layernorm=False):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(layers):
            self.layers.append(torch.nn.Linear(
                input_size if i == 0 else hidden_size,
                output_size if i == layers - 1 else hidden_size,
            ))
            if i != layers - 1:
                self.layers.append(torch.nn.ReLU())
        if layernorm:
            self.layers.append(torch.nn.LayerNorm(output_size))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(0, 1 / math.sqrt(layer.in_features))
                layer.bias.data.fill_(0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class EdgeConv2(torch.nn.Module):
    def __init__(self,hidden_size,layers) -> None:
        super().__init__()
        self.layer = EdgeConv(nn=MLP(2*hidden_size,hidden_size,hidden_size,layers),aggr='sum')
    def forward(self,x,index):
        x = x + self.layer(x = x,edge_index = index)
        return x 

class GNN_edgeConv(torch.nn.Module):
    def __init__(
        self,
        hidden_size = 128,
        n_mp_layers = 5, # number of GNN layers
        node_feat = 1,
        window_size = 5, # the model looks into W frames before the frame to be predicted
    ):
        super().__init__()
        self.window_size = window_size
        self.node_in = MLP(node_feat * window_size, hidden_size, hidden_size, 3)
        self.node_out = MLP(hidden_size, hidden_size, node_feat, 3, layernorm=False)
        self.n_mp_layers = n_mp_layers
        self.layers = torch.nn.ModuleList([EdgeConv2(hidden_size,3) 
        for _ in range(n_mp_layers)])

    def forward(self, x,index):

        node_feature = self.node_in(x)
        # stack of GNN layers
        for i in range(self.n_mp_layers):
            node_feature= self.layers[i](node_feature,index)
        # post-processing
        out = self.node_out(node_feature)
        return out

def Initialize(X,Y):
    
    pos_x = torch.tensor(np.array(X),dtype=torch.float64)
    pos_y = torch.tensor(np.array(Y),dtype=torch.float64)
    positions= torch.stack((pos_x,pos_y),axis=1)
    edge_index = pyg.nn.knn_graph(positions,k=10)

    #Initialize_model
    model = GNN_edgeConv(hidden_size=32,n_mp_layers = 5,node_feat=1,window_size=5)
    file = './python_func/last_model.pt'
    checkpoint = torch.load(file,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    device = torch.device('cpu')
    model.to(device,dtype=float)
    model.eval()
    return edge_index,model

idx,model = Initialize(X,Y)

