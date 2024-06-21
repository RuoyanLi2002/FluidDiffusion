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

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
alpha = 0.015
batch_size = 256
lr = 1e-4
num_epoch = 100

ddpm = DDPM(batch_size = batch_size, device = device)
ddpm.to(device)

criterion = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(ddpm.parameters(), lr=lr)

if os.path.isfile("graph_data.pkl"):
    with open('graph_data.pkl', 'rb') as f:
        data_list = pickle.load(f)
else:      
    print("Loading Data")
    with open('train.pkl', 'rb') as f:
        particle_pos, mask = pickle.load(f)

    def create_edge_index(positions):
        tree = neighbors.KDTree(positions)
        receivers_list = tree.query_radius(positions, r=alpha)
        num_nodes = len(positions)
        senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
        receivers = np.concatenate(receivers_list, axis=0)

        # Remove self edges.
        mask = senders != receivers
        senders = senders[mask]
        receivers = receivers[mask]

        senders = torch.from_numpy(senders)
        receivers = torch.from_numpy(receivers)
        edge_index = torch.stack([senders, receivers], dim=0)
        
        return edge_index

    def create_graph(particle_pos, mask):
        global_mean = torch.tensor([0.4992, 0.2964])
        global_std = torch.tensor([0.2409, 0.1978])

        edge_index = create_edge_index(particle_pos)

        x = torch.tensor(particle_pos, dtype=torch.float)
        x = (x - global_mean) / global_std
        return Data(x=x, edge_index=edge_index, y=mask)

    print("Creating Graph")
    data_list = [create_graph(particle_pos[i], mask[i]) for i in tqdm(range(len(particle_pos)), desc="Creating Graphs")]

    with open('graph_data.pkl', 'wb') as file:
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
        perturbed_x, epsilon, pred_epsilon = ddpm(data)
        loss = criterion(pred_epsilon, epsilon)

        mask = mask.unsqueeze(1).repeat(1, 2)
        masked_loss = loss * mask
        final_loss = masked_loss.mean()
        
        final_loss.backward()
        optimizer.step()
        
        total_loss += final_loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss}")

torch.save(ddpm.state_dict(), 'model_no_norm.pth')