import pickle
import numpy as np
import torch

file_path = 'WaterRamps/test.pkl'
# ramp 3 water 5
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# max_num_water_particle = 0
# max_num_boundary_particle = 0
max_num_water_particle = 2227
max_num_boundary_particle = 237

# for element in data:
#     particle_type = np.array(element[0]["particle_type"])
#     position = element[1]["position"]

#     water_indices = np.where(particle_type == 5)[0]
#     boundary_indices = np.where(particle_type == 3)[0]
#     water_particle = position[:, water_indices, :]
#     boundary_particle = position[:, boundary_indices, :]

#     num_water_particle = water_particle.shape[1]
#     num_boundary_particle = boundary_particle.shape[1]

#     if num_water_particle > max_num_water_particle:
#         max_num_water_particle = num_water_particle

#     if num_boundary_particle > max_num_boundary_particle:
#         max_num_boundary_particle = num_boundary_particle
# print(f"max_num_water_particle: {max_num_water_particle}")
# print(f"max_num_boundary_particle: {max_num_boundary_particle}")

ls_water_particle = []
ls_boundary_particle = []
ls_mask = []

sliding_window = 50
fix_sequence_length = 100
for element in data:
    particle_type = np.array(element[0]["particle_type"])
    position = element[1]["position"]
    sequence_length = position.shape[0]
    water_indices = np.where(particle_type == 5)[0]
    boundary_indices = np.where(particle_type == 3)[0]
    water_particle = position[:, water_indices, :]
    boundary_particle = position[:, boundary_indices, :]

    num_water_particle = water_particle.shape[1]
    num_boundary_particle = boundary_particle.shape[1]

    start = 0
    end = start + fix_sequence_length
    while end < sequence_length:
        temp_water_particle = water_particle[start:end, :, :]
        temp_boundary_particle = boundary_particle[start:end, :, :]

        padded_water_particle = np.zeros((fix_sequence_length, max_num_water_particle, 2))
        padded_boundary_particle = np.zeros((fix_sequence_length, max_num_boundary_particle, 2))
        mask = np.zeros((fix_sequence_length, max_num_water_particle, 2))

        padded_water_particle[:, :num_water_particle, :] = temp_water_particle
        padded_boundary_particle[:, :num_boundary_particle, :] = temp_boundary_particle
        mask[:, :num_water_particle, :] = 1

        ls_water_particle.append(torch.from_numpy(padded_water_particle))
        ls_boundary_particle.append(torch.from_numpy(padded_boundary_particle))
        ls_mask.append(torch.from_numpy(mask))

        start = start + sliding_window
        end = start + fix_sequence_length

print(len(ls_water_particle))
print(len(ls_boundary_particle))
print(len(ls_mask))

with open('WaterRamps/test_split.pkl', 'wb') as file:
    pickle.dump([ls_water_particle, ls_boundary_particle, ls_mask], file)