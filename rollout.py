import os
import pickle
import numpy as np
import torch

from model import DDPM


file_path = '/home/liruoyan2002/WaterRamps/train.pkl'
# ramp 3 water 5
with open(file_path, 'rb') as file:
    data = pickle.load(file)

for element in data:
    particle_type = np.array(element[0]["particle_type"])
    position = element[1]["position"]
    position = torch.tensor(position)
    particle_type = torch.tensor(particle_type)

    if (particle_type == 3).any():
        break
    else:
        print("No Ramp")

boundary = position[0, particle_type == 3, :]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ddpm = DDPM(batch_size = 1, device = device)
ddpm.to(device)
checkpoint = torch.load('model_no_norm.pth')
ddpm.load_state_dict(checkpoint)


first_frame = position[0].unsqueeze(0)

rollout = [first_frame]
for i in range(1, position.shape[0]):
    print(f"i: {i}")
    x_t = rollout[i-1].squeeze(0)
    x_t = x_t.to(device)
    result = ddpm.sample(x_t)
    
    result = result.detach().cpu()

    result[particle_type == 3, :] = boundary
    
    rollout.append(result.unsqueeze(0))

print(len(rollout))

rollout = torch.cat(rollout, dim = 0)
print(f"rollout: {rollout.shape}")
print(f"position: {position.shape}")

rollout_data = {"ground_truth_rollout": position, "predicted_rollout": rollout, "particle_types": particle_type}

with open('gnn_rollout_data.pkl', 'wb') as file:
    pickle.dump(rollout_data, file)