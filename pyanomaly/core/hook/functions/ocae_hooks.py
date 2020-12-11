"""
@author:  Yuhao Cheng
@contact: yuhao.cheng[at]outlook.com
"""
import os
import pickle
import cv2
import torch
import mmcv
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
from pyanomaly.datatools.evaluate.utils import psnr_error, oc_score
from pyanomaly.core.utils import multi_obj_grid_crop, frame_gradient, flow_batch_estimate, get_batch_dets, tensorboard_vis_images, save_score_results
from pyanomaly.core.other.kmeans import kmeans, kmeans_predict
# from lib.datatools.evaluate import eval_api
from ..abstract import HookBase, EvaluateHook

# from sklearn.cluster import KMeans
# from kmeans_pytorch import kmeans, kmeans_predict
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
# from sklearn.cluster import KMeans
# from sklearn.metrics import calinski_harabaz_score
from sklearn.preprocessing import MultiLabelBinarizer
try:
    from sklearn.externals import joblib
except:
    import joblib

# HOOKS=['ClusterHook', 'OCEvaluateHook']
from ..hook_registry import HOOK_REGISTRY

__all__ = ['ClusterHook', 'OCEvaluateHook']

@HOOK_REGISTRY.register()
class ClusterHook(HookBase):
    def after_step(self, current_step):
        # import ipdb; ipdb.set_trace()
        if current_step % self.trainer.config.TRAIN.eval_step == 0 and current_step!= 0:
            self.trainer.logger.info('Start clsuter the feature')
            frame_num = self.trainer.config.DATASET.train.clip_length
            frame_step = self.trainer.config.DATASET.train.clip_step
            feature_record = []
            for video_name in self.trainer.cluster_dataset_keys:
                dataset = self.trainer.cluster_dataset_dict[video_name]
                data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
                # import ipdb; ipdb.set_trace()
                for test_input, anno, meta in data_loader:
                    future = data[:, :, 2, :, :].cuda() # t+1 frame 
                    current = data[:, :, 1, :, :].cuda() # t frame
                    past = data[:, :, 0, :, :].cuda() # t frame
                    bboxs = get_batch_dets(self.trainer.Detector, current)
                    for index, bbox in enumerate(bboxs):
                        # import ipdb; ipdb.set_trace()
                        if bbox.numel() == 0:
                            # import ipdb; ipdb.set_trace()
                            # bbox = torch.zeros([1,4])
                            bbox = bbox.new_zeros([1,4])
                            # print('NO objects')
                            # continue
                        # import ipdb; ipdb.set_trace()
                        current_object, _ = multi_obj_grid_crop(current[index], bbox)
                        future_object, _ = multi_obj_grid_crop(future[index], bbox)
                        future2current = torch.stack([future_object, current_object], dim=1)

                        past_object, _ = multi_obj_grid_crop(past[index], bbox)
                        current2past = torch.stack([current_object, past_object], dim=1)

                        _, _, A_input = frame_gradient(future2current)
                        A_input = A_input.sum(1)
                        _, _, C_input = frame_gradient(current2past)
                        C_input = C_input.sum(1)
                        A_feature, _, _ = self.trainer.A(A_input)
                        B_feature, _, _ = self.trainer.B(current_object)
                        C_feature, _, _ = self.trainer.C(C_input)
                        
                        A_flatten_feature = A_feature.flatten(start_dim=1)
                        B_flatten_feature = B_feature.flatten(start_dim=1)
                        C_flatten_feature = C_feature.flatten(start_dim=1)
                        ABC_feature = torch.cat([A_flatten_feature, B_flatten_feature, C_flatten_feature], dim=1).detach()
                        # import ipdb; ipdb.set_trace()
                        ABC_feature_s = torch.chunk(ABC_feature, ABC_feature.size(0), dim=0)
                        # feature_record.extend(ABC_feature_s)
                        for abc_f in ABC_feature_s:
                            temp = abc_f.squeeze(0).cpu().numpy()
                            feature_record.append(temp)
                        # import ipdb; ipdb.set_trace()
                self.trainer.logger.info(f'Finish the video:{video_name}')
            self.trainer.logger.info(f'Finish extract feature, the sample:{len(feature_record)}')
            device = torch.device('cuda:0')
            cluster_input = torch.from_numpy(np.array(feature_record))
            # cluster_input = np.array(feature_record)
            time = mmcv.Timer()
            # import ipdb; ipdb.set_trace()
            cluster_centers = cluster_input.new_zeros(size=[self.trainer.config.TRAIN.cluster.k, 3072])
            cluster_score = 0.0
            cluster_model = None
            for _ in range(1):
                # model = KMeans(n_clusters=self.trainer.config.TRAIN.cluster.k, init='k-means++',n_init=10, algorithm='full',max_iter=300).fit(cluster_input)
                # labels = model.labels_
                # temp = calinski_harabaz_score(cluster_input, labels)
                # if temp > cluster_score:
                    # cluster_model = model
                # print(f'the temp score is {temp}')
                cluster_ids_x, cluster_center = kmeans(X=cluster_input, num_clusters=self.trainer.config.TRAIN.cluster.k, distance='euclidean', device=device)
                cluster_centers += cluster_center
            # import ipdb; ipdb.set_trace()
            # cluster_centers =  cluster_centers / 10
            # model.fit(cluster_input)
            # pusedo_labels = model.predict(cluster_input)
            pusedo_labels = kmeans_predict(cluster_input, cluster_centers, 'euclidean', device=device).detach().cpu().numpy()
            # pusedo_labels = cluster_model.labels_
            print(f'The cluster time is :{time.since_start()/60} min')
            # import ipdb; ipdb.set_trace()
            # pusedo_labels = np.split(pusedo_labels, pusedo_labels.shape[0], 0)

            pusedo_dataset = os.path.join(self.trainer.config.TRAIN.pusedo_data_path, 'pusedo')
            if not os.path.exists(pusedo_dataset):
                os.mkdir(pusedo_dataset)
            
            np.savez_compressed(os.path.join(pusedo_dataset, f'{self.trainer.config.DATASET.name}_dummy.npz'), data=cluster_input, label=pusedo_labels)
            print(f'The save time is {time.since_last_check() / 60} min')
            # binary_labels = MultiLabelBinarizer().fit_transform(pusedo_labels)
            # self.trainer.ovr_model = OneVsRestClassifier(LinearSVC(random_state = 0)).fit(cluster_input,binary_labels)
            # self.trainer.ovr_model = OneVsRestClassifier(LinearSVC(random_state = 0), n_jobs=16).fit(cluster_input, pusedo_labels)
            self.trainer.ovr_model = self.trainer.ovr_model.fit(cluster_input, pusedo_labels)
            # self.trainer.saved_model['OVR'] = self.trainer.ovr_model
            print(f'The train ovr: {time.since_last_check() / 60} min')
            joblib.dump(self.trainer.ovr_model, self.trainer.ovr_model_path)
            # import ipdb; ipdb.set_trace()
            

class OCEvaluateHook(EvaluateHook):    
    def evaluate(self, current_step):
        '''
        Evaluate the results of the model
        !!! Will change, e.g. accuracy, mAP.....
        !!! Or can call other methods written by the official
        '''
        self.trainer.set_requires_grad(self.trainer.A, False)
        self.trainer.set_requires_grad(self.trainer.B, False)
        self.trainer.set_requires_grad(self.trainer.C, False)
        self.trainer.set_requires_grad(self.trainer.Detector, False)
        self.trainer.A.eval()
        self.trainer.B.eval()
        self.trainer.C.eval()
        self.trainer.Detector.eval()
        
        frame_num = self.trainer.config.DATASET.test_clip_length
        tb_writer = self.trainer.kwargs['writer_dict']['writer']
        global_steps = self.trainer.kwargs['writer_dict']['global_steps_{}'.format(self.trainer.kwargs['model_type'])]
        score_records = []
        # psnr_records = []
        total = 0
        random_video_sn = torch.randint(0, len(self.trainer.test_dataset_keys), (1,))
        # random_video_sn = 0
        for sn, video_name in enumerate(self.trainer.test_dataset_keys):
            # _temp_test_folder = os.path.join(self.testing_data_folder, dir)
            # need to improve
            # dataset = AvenueTestOld(_temp_test_folder, clip_length=frame_num)
            dataset = self.trainer.test_dataset_dict[video_name]
            len_dataset = dataset.pics_len
            test_iters = len_dataset - frame_num + 1
            test_counter = 0
            # feature_record = []
        
            data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
            # import ipdb; ipdb.set_trace()
            scores = np.empty(shape=(len_dataset,),dtype=np.float32)
            # for test_input, _ in data_loader:
            random_frame_sn = torch.randint(0, len_dataset,(1,))
            for frame_sn, (test_input, anno, meta) in enumerate(data_loader):
                feature_record_object = []
                future = test_input[:, :, 2, :, :].cuda()
                current = test_input[:, :, 1, :, :].cuda()
                past = test_input[:, :, 0, :, :].cuda()
                bboxs = get_batch_dets(self.trainer.Detector, current)
                for index, bbox in enumerate(bboxs):
                    # import ipdb; ipdb.set_trace()
                    if bbox.numel() == 0:
                        bbox = bbox.new_zeros([1,4])
                        # print('NO objects')
                        # continue
                    current_object, _ = multi_obj_grid_crop(current[index], bbox)
                    future_object, _ = multi_obj_grid_crop(future[index], bbox)
                    future2current = torch.stack([future_object, current_object], dim=1)

                    past_object, _ = multi_obj_grid_crop(past[index], bbox)
                    current2past = torch.stack([current_object, past_object], dim=1)

                    _, _, A_input = frame_gradient(future2current)
                    A_input = A_input.sum(1)
                    _, _, C_input = frame_gradient(current2past)
                    C_input = C_input.sum(1)
                    A_feature, temp_a, _ = self.trainer.A(A_input)
                    B_feature, temp_b, _ = self.trainer.B(current_object)
                    C_feature, temp_c, _ = self.trainer.C(C_input)

                    # import ipdb; ipdb.set_trace()
                    if sn == random_video_sn and frame_sn == random_frame_sn:
                        vis_objects = OrderedDict({
                            'eval_oc_input_a': A_input.detach(),
                            'eval_oc_output_a': temp_a.detach(),
                            'eval_oc_input_b': current_object.detach(),
                            'eval_oc_output_b':  temp_b.detach(),
                            'eval_oc_input_c': C_input.detach(),
                            'eval_oc_output_c': temp_c.detach(),
                        })
                        tensorboard_vis_images(vis_objects, tb_writer, global_steps, normalize=self.trainer.normalize.param['val'])

                    A_flatten_feature = A_feature.flatten(start_dim=1)
                    B_flatten_feature = B_feature.flatten(start_dim=1)
                    C_flatten_feature = C_feature.flatten(start_dim=1)
                    ABC_feature = torch.cat([A_flatten_feature, B_flatten_feature, C_flatten_feature], dim=1).detach()
                    ABC_feature_s = torch.chunk(ABC_feature, ABC_feature.size(0), dim=0)

                    for abc_f in ABC_feature_s:
                        temp = abc_f.squeeze(0).cpu().numpy()
                        feature_record_object.append(temp)
                
                predict_input = np.array(feature_record_object)
                self.trainer.ovr_model = joblib.load(self.trainer.ovr_model_path)
                g_i = self.trainer.ovr_model.decision_function(predict_input) # the svm score of each object in one frame
                frame_score = oc_score(g_i)

                # test_psnr = psnr_error(g_output, test_target)
                # test_psnr = test_psnr.tolist()
                scores[test_counter+frame_num-1] = frame_score

                test_counter += 1
                total+=1
                if test_counter >= test_iters:
                    scores[:frame_num-1]=scores[frame_num-1]
                    score_records.append(scores)
                    print(f'finish test video set {video_name}')
                    break
        
        self.trainer.pkl_path = save_score_results(self.trainer.config, self.trainer.logger, verbose=self.trainer.verbose, config_name=self.trainer.config_name, current_step=current_step, time_stamp=self.trainer.kwargs["time_stamp"],score=score_records)
        results = self.trainer.evaluate_function(self.trainer.pkl_path, self.trainer.logger, self.trainer.config)
        self.trainer.logger.info(results)
        tb_writer.add_text('AUC of ROC curve', f'AUC is {results.auc:.5f}',global_steps)
        return results.auc

