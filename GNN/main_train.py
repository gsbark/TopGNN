import os
import shutil 
import torch
import math
from torch_geometric.loader import DataLoader
from gnn_utils import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time 
import matplotlib.pyplot as plt
from itertools import product
from torch.utils.data import random_split
from sklearn.metrics import f1_score,accuracy_score


#Initiallize
###########################
torch.manual_seed(0)
# #Results path
shutil.rmtree('./runs', ignore_errors=True)
#Saved models path
shutil.rmtree('./Models', ignore_errors=True)
os.makedirs('./Models')
# ###########################

#Get Train/Test Dataset 
def read_data(windows=5,split=0.8,train_batch=4,eval_batch=4):
    
    dataset = Get_dataset(windows=windows) 
    split = 0.8

    num_total_samples = dataset.len()
    num_samples_train = int(math.ceil(split*num_total_samples))
    num_samples_test = int(math.floor(round(1-split,2)*num_total_samples))

    train_dataset,eval_dataset = random_split(dataset,[num_samples_train,num_samples_test])
    train_loader = DataLoader(train_dataset, batch_size = train_batch, shuffle = True)
    eval_loader = DataLoader(eval_dataset, batch_size = eval_batch, shuffle = True)
    return train_loader,eval_loader

train_loader,test_loader = read_data(windows=5,split=0.8,train_batch=4,eval_batch=4)

def run_train(epochs = 500,mlp_hidden=128,mp_layers=2,time_windows=5,loss_arg='BCE',module="Inter"):

    com = f'__mlp_hidden={mlp_hidden}_mp_layer={mp_layers}_Time_windows={time_windows}_GNN_module={module}_Loss={loss_arg}'
    os.makedirs('./Models/'+ com)
    save_path = './Models/' + com + "/last_model"

    #Model Configuration
    #Inputs : Density
    if module=="Inter":
        model = InteractionGNN(hidden_size = mlp_hidden,n_mp_layers = mp_layers,node_feat=1,window_size=time_windows)
    if module=="Edge_conv":
        model = GNN_edgeConv(hidden_size = mlp_hidden,n_mp_layers = mp_layers,node_feat=1,window_size=time_windows)
    
    #pytorch_total_params = sum(p.numel() for p in model.parameters())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Training on {}'.format(device))

    #Optimizer Configuration
    model.to(device,dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(),lr = 1e-3,)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.95)

    if loss_arg =='BCE':
        loss_fun = BCE_const
        metrics = metrics_binary
    
    if loss_arg =='MSE':
        loss_fun = MSE_const

    if loss_arg =='MAE':
        loss_fun = MAE_const
    
    best_loss = torch.inf
    time_start = time.time()

    for epoch in range(epochs):
        epoch_num = epoch +1 
        model.train()
        train_loss = 0
        train_accuracy = 0 
        train_f1_score = 0 
        for data in tqdm(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            pred = model(data.x,data.edge_index)
            loss = loss_fun(pred.squeeze(1),data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if loss_arg =='BCE':
                acc,f1_sc = metrics(pred.squeeze(1),data.y)
                train_accuracy += acc
                train_f1_score += f1_sc
        
        train_accuracy /= len(train_loader)
        train_f1_score /= len(train_loader)
        train_loss /= len(train_loader)

        test_loss = 0
        test_accuracy = 0
        test_f1_score = 0 

        #Evaluate the model
        model.eval()
        for test_data in test_loader:
            test_data = test_data.to(device)
            with torch.no_grad(): 
                t_pred = model(test_data.x,test_data.edge_index)
            loss_t = loss_fun(t_pred.squeeze(1),test_data.y)
            test_loss += loss_t.item()

            if loss_arg =='BCE':
                acc_t,f1_sc_t = metrics(t_pred.squeeze(1),test_data.y)
                test_accuracy += acc_t
                test_f1_score +=f1_sc_t
        
        test_loss /=len(test_loader)
        test_accuracy /= len(test_loader)
        test_f1_score /= len(test_loader)

        if test_loss < best_loss:
            torch.save({
                'epoch': epoch_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train loss': train_loss,
                'test loss':test_loss
                }, save_path)
            saved_accuracy = test_accuracy
            best_loss = test_loss

        scheduler.step()
        print('Epoch: %d | Train loss = %.8f| Test loss = %.8f| Train Acc = %.8f| Test Acc = %.8f|'%(epoch_num, train_loss,test_loss,train_accuracy,test_accuracy))
        print("Learning_rate :",optimizer.param_groups[0]['lr'])
        print("time elapsed --- %.2f min ---" % ((time.time() - time_start)/60))
        
        writer.add_scalar(tag='Loss/Train_loss',scalar_value = train_loss,global_step = epoch_num,walltime=(time.time() - time_start)/60)
        writer.add_scalar(tag='Loss/Test_loss',scalar_value = test_loss,global_step = epoch_num,walltime=(time.time() - time_start)/60)

        if loss_arg =='BCE':
            writer.add_scalar(tag='Metrics/Train_acc',scalar_value = train_accuracy,global_step = epoch_num,walltime=(time.time() - time_start)/60)
            writer.add_scalar(tag='Metrics/Test_acc',scalar_value = test_accuracy,global_step = epoch_num,walltime=(time.time() - time_start)/60)
            writer.add_scalar(tag='Metrics/Train_f1score',scalar_value = train_f1_score,global_step = epoch_num,walltime=(time.time() - time_start)/60)
            writer.add_scalar(tag='Metrics/Test_f1score',scalar_value = test_f1_score,global_step = epoch_num,walltime=(time.time() - time_start)/60)
        writer.flush()
    return best_loss,saved_accuracy

if __name__=='__main__':
    #Model parameters
    parameters = dict(
        mlp_hidden = [32],
        mp_layers = [5],
        model = ["Edge_conv"],
        time_windows = [5],
        loss_type = ['BCE'])

    param_values = [v for v in parameters.values()]

    for mlp_hidden,mp_layers,model,time_windows,loss_type in product(*param_values):
        print("-"*30)
        print("Model_parameters :","mlp_hidden :", mlp_hidden,"mp_layers :", mp_layers,"Time_windows:",time_windows,"GNN_module:" ,model,"Loss:" ,loss_type)
        print("-"*30)

        com = f'__mlp_hidden={mlp_hidden}_mp_layer={mp_layers}_Time_windows={time_windows}_GNN_module={model}_Loss={loss_type}'
        writer = SummaryWriter(comment=com)
        total_loss,accuracy = run_train(epochs=200,mlp_hidden=mlp_hidden,mp_layers=mp_layers,time_windows = time_windows,module = model,loss_arg=loss_type)

        writer.add_hparams({"mlp_hidden":mlp_hidden,
                            "mp_layers": mp_layers,
                            "time_windows":time_windows,
                            "Model":model,
                            'Loss':loss_type},
                            {"Loss":total_loss,
                            "Accuracy":accuracy})
        writer.flush()
    writer.close()

