import pickle
import numpy as np
from sklearn import neighbors
import torch

file_path = '/home/liruoyan2002/ramp/WaterRamps/train.pkl'
# ramp 3 water 5
with open(file_path, 'rb') as file:
    data = pickle.load(file)

pos_ls = []
mask_ls = []

alpha = 0.015

def create_edge_index(x_t):
    num_particles = x_t.shape[0]
    distances = torch.cdist(x_t, x_t, p=2)
    mask = (distances < alpha) & (distances > 0)

    edge_index = torch.nonzero(mask).t().contiguous()

    return edge_index

def my_create_edge_index(x_t):
    num_particles = x_t.shape[0]

    sender = []
    receiver = []
    
    for i in range(num_particles):
        for j in range(num_particles):
            if i != j and torch.sqrt(torch.sum((x_t[i, ] - x_t[j, ])**2)) < alpha:
                sender.append(i)
                receiver.append(j)

    sender = torch.tensor(sender).unsqueeze(0)
    receiver = torch.tensor(receiver).unsqueeze(0)

    edge_index = torch.cat([sender, receiver], dim = 0)

    return edge_index

def correct_create_edge_index(positions):
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


for element in data:
    particle_type = np.array(element[0]["particle_type"])
    position = element[1]["position"]
    particle_type = torch.tensor(particle_type)

    for i in range(position.shape[0]):
        pos_ls.append(position[i, :, :])

        mask = (particle_type == 5).int()
        mask_ls.append(mask)


print(len(pos_ls))
print(len(mask_ls))


with open('train.pkl', 'wb') as file:
    pickle.dump([pos_ls, mask_ls], file)