import os
import sys
import cv2
sys.path.append('../../')
print(sys.path)
import json
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import torchvision.transforms.functional as tf
from tqdm import tqdm
from lib.utils.image_ops import image_gradient


def make_objects_db(data_path, split='training',det_threshold=0.95, time_file='./training_3.json', verbose='none'):
    """
    Make the database based on the detections
    Args:
        data_path: e.g. 'data/shanghaitech/normal'
        det_threshold: 0.5
    """
    original_path = Path(data_path) / split
    detection_path = original_path / 'detection_cascade'
    images_path = original_path / 'frames'
    print(original_path)
    final_path = original_path / 'objects'
    # temp_folder = './temp'
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    final = dict()
    with open(time_file, 'r') as f:
        temp = json.load(f)

    det_results = sorted(os.listdir(detection_path))
    pbar = tqdm(range(len(det_results)))
    # count = 0
    for frame_det in det_results:
        pbar.set_description('Processing: {}'.format(frame_det))
        video_name = '_'.join(frame_det.split('.')[0].split('_')[0:-1])
        image_name = frame_det.split('.')[0].split('_')[-1]
        npy_file = detection_path / frame_det
        detections = np.load(npy_file)
        finals = _produce_detection(detections[0], detections[1], detections[2], detections[3], detections[7])
        # finals = _produce_detection(detections[0])
        for index, det in enumerate(finals):
            if det[4] <= det_threshold:
                continue
            current_frame = frame_det.split('.')[0]
            back_frame = temp[current_frame]['back']
            front_frame = temp[current_frame]['front']
            object_c = _crop_det_cv2(images_path / video_name / (image_name + '.jpg'), det)
            object_b = _crop_det_cv2(images_path / video_name / (back_frame + '.jpg'), det)
            object_f = _crop_det_cv2(images_path / video_name / (front_frame + '.jpg'), det)
            try:
                back_gradient_x, back_gradient_y = image_gradient(object_b.unsqueeze_(0).unsqueeze_(0))
            except Exception as err:
                print(err)
                import ipdb; ipdb.set_trace()
            back_gradient = torch.cat([back_gradient_x, back_gradient_y], dim=1).squeeze_(0)
            front_gradient_x, front_gradient_y = image_gradient(object_f.unsqueeze_(0).unsqueeze_(0))
            front_gradient = torch.cat([front_gradient_x, front_gradient_y], dim=1).squeeze_(0)
            # import ipdb; ipdb.set_trace()
            final[frame_det.split('.')[0] + '#' +str(index)] = dict()
            final[frame_det.split('.')[0] + '#' +str(index)]['current'] = object_c
            final[frame_det.split('.')[0] + '#' +str(index)]['back_gradient'] = back_gradient
            final[frame_det.split('.')[0] + '#' +str(index)]['front_gradient'] = front_gradient
        # count += 1
        pbar.update(1)
        # if count > 20:
        #     break
    try:
        final_name = final_path / (split + '_' + verbose + '.pt')
        torch.save(final, final_name)
    except Exception as err:
        print(err)         


def _produce_detection(*args):
        # make the empty detection format
        dummy = np.array([0,0,0,0,0], dtype=np.float64) 
        lm = lambda x: x if x.size > 0 else np.array([dummy])
        init = np.array([dummy])
        for detect in args:
            # pad the None object
            new_detect = lm(detect)
            # make them in one array
            init = np.concatenate((init, new_detect))
        # filter the not empty ones 
        f = filter(lambda x : (x != np.array(dummy)).any(), init)
        new_init = [x for x in f]

        # deal with the situation of no detection
        if new_init == []:
            new_init = np.array([dummy])
        
        return new_init

def _crop_det(image_path, det, mode='gray'):
    '''
    Args:
        mode: 'gray' or 'rgb'
    '''
    img = Image.open(image_path).convert('L')
    xmin = det[0]
    ymin = det[1]
    xmax = det[2]
    ymax = det[3]
    height = ymax - ymin
    width = xmax -xmin
    
    temp_obj = tf.crop(img, int(ymin), int(xmin), int(height), int(width))    # ymin, xmin, height, width
    obj = tf.to_tensor(temp_obj)
    return obj

def _crop_det_cv2(image_path, det):
    '''
    Args:
        mode: 'gray' or 'rgb'
    '''
    
    print(image_path)
    image_path = str(image_path)
    try:
        img = cv2.imread(image_path, 0)
    except:
        print('in _crop_cv2')
        import ipdb; ipdb.set_trace()
    xmin = int(det[0])
    ymin = int(det[1])
    xmax = int(det[2])
    ymax = int(det[3])
    height = ymax - ymin
    width = xmax -xmin
    
    # temp_obj = tf.crop(img, int(ymin), int(xmin), int(height), int(width))    # ymin, xmin, height, width
    temp_obj = img[ymin:ymax, xmin:xmax]  # ymin, xmin, height, width
    obj = torch.from_numpy(temp_obj)
    return obj
    
def _get_gradient(image_1, image_2):
        '''
        images_1 - images_2 
        '''
        gradient = lambda x, y: x - y
        # images_pair = zip(images_1, images_2)
        # gs = list()  # gs = gradients of the images
        # for x, y in images_pair:
        im_1 = tf.to_tensor(image_1)
        im_2 = tf.to_tensor(image_2)
        gs = gradient(im_1,im_2)
            # gs.append(temp)
        
        return gs

def make_global_db(data_path, split='testing'):
    """
    Make the database based on the detections
    Args:
        data_path: e.g. 'data/shanghaitech/normal'
    """
    original_path = Path(data_path)
    final_path = original_path / 'global'
    images_path = original_path / 'frames'
    # temp_folder = './temp'
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    final = dict()
    # with open(time_file, 'r') as f:
    #     temp = json.load(f)

    video_list = sorted(os.listdir(images_path))
    pbar = tqdm(range(len(video_list)))
    # count = 0
    
    for video in video_list:
        pbar.set_description('Processing: {}'.format(video))
        video_path = images_path / video
        # finals = _produce_detection(detections[0])
        images_list = sorted(os.listdir(video_path))
        for index, image in enumerate(images_list):
            temp = Image.open(video_path / image).convert('L')
            final[video + '#' +str(index)] = temp
        # count += 1
        pbar.update(1)
        # if count > 20:
        #     break
    
    final_name = final_path / (split + '.pt')
    torch.save(final, final_name)         

def decide_back_front(dataset_path, verbose='testing_vad', duration=3):
    '''
    decide the back and front, save in json
    Args:
        dataset_path: e.g. './data/shanghaitech/training/frames'
        duration: step, e.g. current-3, current+3
    '''
    video_path = Path(dataset_path)
    video_list = sorted(os.listdir(video_path))
    final = dict()
    for video in video_list:
        frames_list = sorted(os.listdir(video_path / video))
        for index, _ in enumerate(frames_list):
            # get the image behind the current frame
            if index - duration <= 0:
                img_b = frames_list[0]
            else:
                img_b = frames_list[index - duration]
            
            # get the image before the current frame
            if index + duration >=  len(frames_list):
                img_f = frames_list[-1]
            else:
                img_f = frames_list[index + duration]
            
            # get the image name
            img_current = video + '_' + frames_list[index].split('.')[0]
            final[img_current] = dict()
            # final[img_current]['back'] = video + '_' + img_b.split('.')[0]
            final[img_current]['back'] = img_b.split('.')[0]
            # final[img_current]['front'] = video + '_' + img_f.split('.')[0]
            final[img_current]['front'] = img_f.split('.')[0]

    with open(verbose+'_'+str(duration) + '.json', 'w') as f:
        json.dump(final, f)
        print('finish') 


def make_objects_box_db(data_path, split='training',det_threshold=0.95, time_file='./training_3.json', verbose='none'):
    """
    Make the database based on the detections
    Args:
        data_path: e.g. 'data/shanghaitech/normal'
        det_threshold: 0.5
    """
    original_path = Path(data_path) / split
    detection_path = original_path / 'detection_cascade'
    # images_path = original_path / 'frames'
    print(original_path)
    final_path = original_path / 'objects'
    # temp_folder = './temp'
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    final = dict()
    with open(time_file, 'r') as f:
        temp = json.load(f)

    det_results = sorted(os.listdir(detection_path))
    pbar = tqdm(range(len(det_results)))
    # count = 0
    for frame_det in det_results:
        pbar.set_description('Processing: {}'.format(frame_det))
        video_name = '_'.join(frame_det.split('.')[0].split('_')[0:-1])
        image_name = frame_det.split('.')[0].split('_')[-1]
        npy_file = detection_path / frame_det
        detections = np.load(npy_file)
        finals = _produce_detection(detections[0], detections[1], detections[2], detections[3], detections[7])
        # finals = _produce_detection(detections[0])
        for index, det in enumerate(finals):
            if det[4] <= det_threshold:
                continue
            current_frame = frame_det.split('.')[0]
            back_frame = temp[current_frame]['back']
            front_frame = temp[current_frame]['front']
            # object_c = _crop_det_cv2(images_path / video_name / (image_name + '.jpg'), det)
            # object_b = _crop_det_cv2(images_path / video_name / (back_frame + '.jpg'), det)
            # object_f = _crop_det_cv2(images_path / video_name / (front_frame + '.jpg'), det)
            # back_gradient_x, back_gradient_y = image_gradient(object_b.unsqueeze_(0).unsqueeze_(0))
            # back_gradient = torch.cat([back_gradient_x, back_gradient_y], dim=1).squeeze_(0)
            # front_gradient_x, front_gradient_y = image_gradient(object_f.unsqueeze_(0).unsqueeze_(0))
            # front_gradient = torch.cat([front_gradient_x, front_gradient_y], dim=1).squeeze_(0)
            # import ipdb; ipdb.set_trace()
            final[frame_det.split('.')[0] + '#' +str(index)] = dict()
            final[frame_det.split('.')[0] + '#' +str(index)]['box'] = det
            final[frame_det.split('.')[0] + '#' +str(index)]['current_frame'] = video_name + '/' + image_name + '.jpg'
            final[frame_det.split('.')[0] + '#' +str(index)]['back_frame'] = video_name + '/' + back_frame + '.jpg'
            final[frame_det.split('.')[0] + '#' +str(index)]['front_frame'] = video_name + '/' + front_frame + '.jpg' 
        # count += 1
        pbar.update(1)
        # if count > 20:
            # break
            # import ipdb; ipdb.set_trace()
    try:
        final_name = final_path / (split + '_' + verbose + '.pt')
        torch.save(final, final_name)
    except Exception as err:
        print(err)         

if __name__ == '__main__':
    # path = '/export/home/chengyh/reproduce/objec-centric/data/VAD/testing/frames'
    # path_objects = '/export/home/chengyh/reproduce/objec-centric/data/VAD/'
    path_objects = '/export/home/chengyh/reproduce/objec-centric/data/SHTech/'
    # decide_back_front(path)
    # make_objects_db(path_objects, split='training', det_threshold=0.6, time_file='./training_3.json', verbose='det0.6_cascade_grad')
    make_objects_box_db(path_objects, split='testing', det_threshold=0.4, time_file='./testing_3.json', verbose='det0.4_cascade_only_box')
    # make_global_db(path_objects)