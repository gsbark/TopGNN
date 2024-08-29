import torch 
import numpy as np 

def run(model,density,index):
    density_tens = np.array(density)
    density_tens = torch.tensor(density_tens)
    sigmoid = torch.nn.Sigmoid()
    with torch.no_grad():
        out = model(density_tens,index)
        probs = sigmoid(out)
    return probs.cpu().numpy()

probs = run(model,density,index)