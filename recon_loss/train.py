import os
import pickle
import numpy as np
from sklearn import neighbors
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from model import DDPM

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
batch_size = 256
lr = 1e-4
num_epoch = 50

ddpm = DDPM(batch_size = batch_size, device = device)
ddpm.to(device)

criterion = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(ddpm.parameters(), lr=lr)

if os.path.isfile("/home/liruoyan2002/ramp2/graph_data_new.pkl"):
    with open('/home/liruoyan2002/ramp2/graph_data_new.pkl', 'rb') as f:
        data_list = pickle.load(f)
else:      
    print("Loading Data")
    with open('/home/liruoyan2002/ramp2/train.pkl', 'rb') as f:
        particle_pos, mask = pickle.load(f)

    def create_graph(particle_pos, mask):
        x = torch.tensor(particle_pos, dtype=torch.float)
        return Data(x=x, edge_index=None, y=mask)

    print("Creating Graph")
    data_list = [create_graph(particle_pos[i], mask[i]) for i in tqdm(range(len(particle_pos)), desc="Creating Graphs")]

    with open('/home/liruoyan2002/ramp2/graph_data_new.pkl', 'wb') as file:
        pickle.dump(data_list, file)

dataloader = DataLoader(data_list, batch_size=batch_size, shuffle=True, drop_last=True)

print("Training")
for epoch in range(num_epoch):
    ddpm.train()
    total_loss = 0
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        mask = data.y
        perturbed_x, x_zeros, pred_x0 = ddpm(data)
        loss = criterion(x_zeros, pred_x0)

        mask = mask.unsqueeze(1).repeat(1, 2)
        masked_loss = loss * mask
        final_loss = masked_loss.mean()
        
        final_loss.backward()
        optimizer.step()
        
        total_loss += final_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

torch.save(ddpm.state_dict(), 'model.pth')