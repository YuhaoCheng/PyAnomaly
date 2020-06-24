import torch 
from torch.utils.data.dataloader import default_collate

import ipdb

def collect_fn(batch):
    """
    image_b, image_a, image, image_f, label, detection_result = batch
    """
    # max_detection = max(list(map(lambda x: len(x[5]), batch)))
    max_detection = max(list(map(lambda x: len(x), batch)))
    for i in range(len(batch)):
        batch[i] = list(batch[i]) # because the element in the batch is a tuple
        dummy = torch.zeros((1,5), dtype=batch[i][5].dtype)
        temp = batch[i][5]
        # make the detection to the same length in order to stack the
        while temp.size(0) < max_detection:
            temp = torch.cat((temp, dummy))
        batch[i][5] = temp
    
    return default_collate(batch)

def collect_fn_local(batch):
    """
    image_b, image_a, image, image_f, crop_objects = batch
    """
    max_detection = max(list(map(lambda x: len(x[4]), batch)))
    for i in range(len(batch)):
        batch[i] = list(batch[i]) # because the element in the batch is a tuple
        dummy = torch.zeros((1,128,64), dtype=batch[i][4][0].dtype)
        temp = batch[i][4]
        # make the detection to the same length in order to stack the
        while temp.size(0) < max_detection:
        # while len(temp) < max_detection:
            temp = torch.cat((temp, dummy))
            # temp.append(dummy)
        batch[i][4] = temp
    
    return default_collate(batch)


