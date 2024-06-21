import numpy as np
from sklearn import neighbors

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv

from IN import InteractionNetwork


class GNNDenoiser(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNDenoiser, self).__init__()
        # self.conv1 = GCNConv(hidden_channels, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # self.conv1 = GATConv(hidden_channels, hidden_channels, heads=8, concat=True)
        # self.conv2 = GATConv(hidden_channels * 8, hidden_channels, heads=4, concat=True)
        # self.conv3 = GATConv(hidden_channels * 4, hidden_channels, heads=1, concat=False)

        self.conv1 = InteractionNetwork(dim_obj = hidden_channels, dim_rel = 1, dim_eff = 128, dim_hidden_obj = 128, dim_hidden_rel = 128)
        self.conv2 = InteractionNetwork(dim_obj = hidden_channels, dim_rel = 1, dim_eff = 128, dim_hidden_obj = 128, dim_hidden_rel = 128)
        self.conv3 = InteractionNetwork(dim_obj = hidden_channels, dim_rel = 1, dim_eff = 128, dim_hidden_obj = 128, dim_hidden_rel = 128)

        self.t_embed = nn.Linear(in_features = 1, out_features = hidden_channels)

        self.encode1 = nn.Linear(in_features = in_channels, out_features = hidden_channels)
        self.encode2 = nn.Linear(in_features = hidden_channels, out_features = hidden_channels)

        self.decode1 = nn.Linear(in_features = hidden_channels, out_features = hidden_channels)
        self.decode2 = nn.Linear(in_features = hidden_channels, out_features = out_channels)

    def forward(self, x, edge_index, timestep, batch_index):
        timestep = timestep.unsqueeze(-1)
        t_embed = F.relu(self.t_embed(timestep.float()))
        t_embed = t_embed[batch_index, ]

        x = F.relu(self.encode1(x))
        x = self.encode2(x)

        x = x + t_embed
        # print(f"-----1----- x: {x.shape}")
        x = F.relu(self.conv1(x, edge_index))
        # print(f"-----2----- x: {x.shape}")
        x = F.relu(self.conv2(x, edge_index))

        x = self.conv3(x, edge_index)
        # print(f"-----3----- x: {x.shape}")

        x = F.relu(self.decode1(x))
        x = self.decode2(x)

        return x

class DDPM(nn.Module):
    def __init__(self, batch_size, timesteps=1000, beta_min=0.0001, beta_max=0.02, device="cpu"):
        super(DDPM, self).__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.timesteps = timesteps
        self.batch_size = batch_size

        self.betas = torch.linspace(start=self.beta_min, end=self.beta_max, steps=timesteps).to(device)
        self.sqrt_betas = torch.sqrt(self.betas)
                                     
        # alpha for forward diffusion
        self.alphas = 1 - self.betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)

        self.model = GNNDenoiser(in_channels=2, hidden_channels=32, out_channels=2)

        self.device = device

    def extract(self, a, t):
        b = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(b, 1)

    def forward_diffusion(self, x_zeros, t, batch_index): 
        epsilon = torch.randn_like(x_zeros).to(self.device)
        
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t)
        sqrt_alpha_bar_expand = sqrt_alpha_bar[batch_index]

        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t)
        sqrt_one_minus_alpha_bar_expand = sqrt_one_minus_alpha_bar[batch_index]
        
        noisy_sample = x_zeros * sqrt_alpha_bar_expand + epsilon * sqrt_one_minus_alpha_bar_expand

        return noisy_sample, epsilon

    def forward(self, data):
        x_zeros = data.x
        edge_index = data.edge_index
        batch_index = data.batch
        
        t = torch.randint(low=0, high=self.timesteps, size=(self.batch_size,)).to(self.device)

        # forward diffusion
        perturbed_x, epsilon = self.forward_diffusion(x_zeros, t, batch_index)
        
        # backward diffusion predict noise
        pred_epsilon = self.model(perturbed_x, edge_index, t, batch_index)
        
        return perturbed_x, epsilon, pred_epsilon

    def denoise_at_t(self, x_t, timestep, t, edge_index):
        if t > 1:
            z = torch.randn_like(x_t).to(self.device)
        else:
            z = torch.zeros_like(x_t).to(self.device)

        batch_index = torch.tensor([0]).repeat_interleave(x_t.shape[0], dim=0).long().to(self.device)
        
        epsilon_pred = self.model(x_t, edge_index, timestep, batch_index)
        
        alpha = self.extract(self.alphas, timestep)
        sqrt_alpha = self.extract(self.sqrt_alphas, timestep)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep)
        sqrt_beta = self.extract(self.sqrt_betas, timestep)
        
        x_t_minus_1 = 1 / sqrt_alpha * (x_t - (1 - alpha) / sqrt_one_minus_alpha_bar * epsilon_pred) + sqrt_beta * z
        
        return x_t_minus_1

    def sample(self, x_t):
        # x_t = torch.randn((x_t.shape[0], x_t.shape[1])).to(self.device)
        edge_index = self.create_edge_index(x_t).to(self.device)
        print(f"edge_index: {edge_index.shape}")
        # denoise from x_T to x_0
        for t in range(999, -1, -1):
            timestep = torch.tensor([t]).long().to(self.device)
            x_t = self.denoise_at_t(x_t, timestep, t, edge_index)
        
        return x_t

    def create_edge_index(self, x_t):
        alpha = 0.015
        positions = x_t.cpu().numpy()

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
