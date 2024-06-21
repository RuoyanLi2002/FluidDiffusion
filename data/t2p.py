import os
import json
import pickle
import functools
import tensorflow as tf
import reading_utils

# ls_folder = ["Sand-3D", "SandRamps", "Water-3D", "WaterDrop", "WaterDrop-XL", "WaterRamps"]
ls_folder = ["WaterRamps"]
# ls_type = ["test", "train", "valid"]
ls_type = ["test"]
part = 0
count = 0
for folder in ls_folder:
    print(folder)
    with open(f'/Volumes/LaCie/learning_to_simulate/Data/{folder}/metadata.json', 'r') as file:
        metadata = json.load(file)

    for type in ls_type:
        data = []

        ds = tf.data.TFRecordDataset([os.path.join(f"/Volumes/LaCie/learning_to_simulate/Data/{folder}", f'{type}.tfrecord')])
        ds = ds.map(functools.partial(
            reading_utils.parse_serialized_simulation_example, metadata=metadata))

        for element in ds.as_numpy_iterator():
            print(element)
            exit()
            # data.append(element)
            # count += 1
            # print(count)

            # if count == 249 or count == 499 or count == 749:
            #     with open(f"/Volumes/LaCie/learning_to_simulate/Data/{folder}/{type}{part}.pkl", 'wb') as file:
            #         pickle.dump(data, file)
            #     data = []
            #     part += 1
            
        with open(f"/Volumes/LaCie/learning_to_simulate/Data/{folder}/{type}.pkl", 'wb') as file:
            pickle.dump(data, file)