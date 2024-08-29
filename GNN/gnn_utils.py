import torch 
import torch_geometric as pyg
from torch_geometric.nn import EdgeConv
from torch_geometric.data import Data,Dataset
from sklearn.metrics import f1_score,accuracy_score
from torch import Tensor
import math
import torch_scatter
import numpy as np 

class CreateDataset(Dataset):
    def __init__(self,density,coords,kNN=10,window_length=5):
        super().__init__()

        self.window_length = window_length
        self.num_samples = density.shape[0]
        self.num_elements = density.shape[1]
        self.num_of_states = density.shape[2] 
        self.windows = []
        
        edge_index = torch.zeros([self.num_samples,2,kNN*self.num_elements],dtype=torch.int64)
        #Create KNN graph for all samples and store them 
        for i in range(self.num_samples):
            edge_index[i,:,:] = pyg.nn.knn_graph(coords[i,:,:],k=kNN)
        
        #Split dataset to sequences based on window length
        for i in range(self.num_samples):
            sequence = {'density_seq':density[i,:,-(window_length+1):-1],
                        'density_target':density[i,:,-1],
                        'edge_index':edge_index[i,:,:],
                        'positions':coords[i,:,:]}
            self.windows.append(sequence)

    def len(self):
        return len(self.windows)
    
    def get(self,idx):

        window = self.windows[idx]
        density_seq = window['density_seq']
        density_tar = window['density_target']
        edge_index = window['edge_index']
        positions = window['positions']
  
        dataset = Data(x = density_seq,
                       edge_index = edge_index,
                       pos = positions,
                       y = density_tar)
        return dataset 

def Get_dataset(windows:int=5,kNN_graph:int=10,dtype:torch.dtype=torch.float32):
    density = torch.tensor(np.load('./Dataset/densities.npy'),dtype=dtype)
    coords = torch.tensor(np.load('./Dataset/coords.npy'),dtype=dtype)
    assert density.shape[-1]-1 >=windows, 'Not enough timesteps in the dataset'
    dataset = CreateDataset(density,coords,kNN=kNN_graph,window_length= windows)
    return dataset 


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers, layernorm=False):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(layers):
            self.layers.append(torch.nn.Linear(
                input_size if i == 0 else hidden_size,
                output_size if i == layers - 1 else hidden_size,dtype=torch.float32
            ))
            if i != layers - 1:
                self.layers.append(torch.nn.ReLU())
        if layernorm:
            self.layers.append(torch.nn.LayerNorm(output_size,dtype=torch.float32))
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

class InteractionNetwork(pyg.nn.MessagePassing):

    def __init__(self, hidden_size, layers):
        super().__init__()
        self.lin_edge = MLP(hidden_size * 2, hidden_size, hidden_size, layers)
        self.lin_node = MLP(hidden_size * 2, hidden_size, hidden_size, layers)

    def forward(self, x: Tensor, edge_index: Tensor, edge_feature: Tensor) -> Tensor:
        edge_out, aggr = self.propagate(edge_index, x=x, edge_feature=edge_feature)
        node_out = self.lin_node(torch.cat((x, aggr), dim=-1))
        edge_out = edge_feature + edge_out
        node_out = x + node_out
        return node_out, edge_out

    def message(self, x_i,edge_feature):
        x = torch.cat((x_i,edge_feature), dim=-1)
        x = self.lin_edge(x)
        return x

    def aggregate(self, inputs, index, dim_size=None):
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        return (inputs, out)


class InteractionGNN(torch.nn.Module):
    def __init__(
        self,
        hidden_size = 128,
        n_mp_layers = 5, # number of GNN layers
        node_feat = 2,
        window_size = 5, # the model looks into W frames before the frame to be predicted
    ):
        super().__init__()
        self.window_size = window_size
        self.hidden = hidden_size
        self.node_in = MLP(node_feat * window_size, hidden_size, hidden_size, 3)
        self.edge_in = MLP(window_size, hidden_size, hidden_size, 3)
        self.node_out = MLP(hidden_size, hidden_size, node_feat, 3, layernorm=False)
        self.n_mp_layers = n_mp_layers
        self.layers = torch.nn.ModuleList([InteractionNetwork(
            hidden_size, 3
        ) for _ in range(n_mp_layers)])

    def forward(self, data):
        #Encode 
        node_feature = self.node_in(data.x)
        edge_feature = self.edge_in(data.edge_attr)
        #Message-Passing layers
        for i in range(self.n_mp_layers):
            node_feature, edge_feature = self.layers[i](node_feature, data.edge_index, edge_feature=edge_feature)
        #Decoder
        out = self.node_out(node_feature)
        return out

class Residual_edgeconv(torch.nn.Module):
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
        self.n_mp_layers = n_mp_layers
        #Encoder 
        self.node_in = MLP(node_feat * window_size, hidden_size, hidden_size, 3)
        self.edge_in = MLP(window_size, hidden_size, hidden_size, 3)
        #Decoder
        self.node_out = MLP(hidden_size, hidden_size, node_feat, 3, layernorm=False)
        #GNN
        self.layers = torch.nn.ModuleList([Residual_edgeconv(hidden_size,3) for _ in range(n_mp_layers)])

    def forward(self,x,edge_index):
        #Encode node features
        node_feature = self.node_in(x)
        #Message-Passing layers
        for i in range(self.n_mp_layers):
            node_feature = self.layers[i](node_feature,edge_index)
        #Decode 
        out = self.node_out(node_feature)
        return out

sigmoid = torch.nn.Sigmoid()


#Loss fun
#____________________________________________

def MSE(y_pred,y_true):
    criterion = torch.nn.MSELoss()
    l1 = criterion(y_pred,y_true)
    return l1

def MSE_const(y_pred,y_true):
    criterion = torch.nn.MSELoss()
    l1 = criterion(y_pred,y_true)
    l2 = torch.abs(torch.sum(y_pred)/y_pred.shape[0] - torch.sum(y_true)/y_pred.shape[0])
    loss = l1+l2
    return loss

def MAE_const(y_pred,y_true):

    crit = torch.nn.L1Loss()
    l1 = crit(y_pred,y_true)
    l2 = torch.abs(torch.sum(y_pred)-torch.sum(y_true))/y_pred.shape[0]
    loss = l1+l2
    
    return loss

def BCE(y_pred,y_true):
    criterion = torch.nn.BCEWithLogitsLoss()

    y_true[y_true>=0.5] = 1  
    y_true[y_true<0.5] = 0
    l1 = criterion(y_pred,y_true)

    return l1

def BCE_const(y_pred,y_true):
    criterion = torch.nn.BCEWithLogitsLoss()
    Y_true = torch.clone(y_true) 
    Y_true[Y_true>=0.5] = 1  
    Y_true[Y_true<0.5] = 0

    l1 = criterion(y_pred,Y_true)
    probs = sigmoid(y_pred)
    pred = (probs >= 0.5).int()
    l2 = torch.abs(torch.sum(probs)/pred.shape[0] - torch.sum(y_true)/y_pred.shape[0])
    loss = l1+l2
    return loss

def metrics_binary(y_pred,y_true):
    Y_true = torch.clone(y_true) 
    Y_true[Y_true>=0.5] = 1  
    Y_true[Y_true<0.5] = 0

    y_probs = sigmoid(y_pred)
    acc = accuracy_score(Y_true.cpu(),y_probs.cpu()>= 0.5)
    f1_sc = f1_score(Y_true.cpu(),y_probs.cpu()>= 0.5)
    return acc, f1_sc

#____________________________________________