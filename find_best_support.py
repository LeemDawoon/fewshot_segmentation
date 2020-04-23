import os
import shutil
import datetime
import argparse
import time
import pickle
import tqdm
import numpy as np
import pandas as pd
import yaml
from tensorboardX import SummaryWriter
import torch
import cv2
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from core.utils.utils import setup_device, setup_seeds

from core.utils.misc import ordered_load, convert_state_dict
from core.utils.loggers import get_logger
from core.augmentations import get_composed_augmentations

from datasets import get_dataset, split_dataset_by_csv, setup_dataloader

from core.utils.metrics import AverageMeter, Metric # SegmentationMetrics

from core.models import get_model
from core.optimizers import get_optimizer
from core.schedulers import get_scheduler
from core.loss import get_loss_function



class Finder(object):    
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger
        self.device = setup_device(cfg.get('gpus', '0'))
        # self.device = 'cpu'
        self.model = None
        self.load_model()
        self.target_img_size = self.cfg['augmentations']['valid_augmentations']['np_scale']
        

    def load_model(self):
        self.logger.info('> load_model')
        self.model = get_model(self.cfg["model"],  self.cfg["data"]["num_classes"])
        self.model.load_state_dict(convert_state_dict(torch.load(self.cfg['test']['checkpoint'])["model_state"]))
        self.model.to(self.device)
        # self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))

    def _preprocess(self, img_path, img_size=224, is_msk=False, is_support=False):
        img = cv2.imread(img_path)
        img = cv2.resize(img,(img_size, img_size), interpolation=cv2.INTER_AREA)
        img = img / 255.
        if is_msk:
            if is_support:
                s_msk_fg = np.where(img > 0.5, 1., 0.)
                s_msk_bg = np.where(img <= 0.5, 1., 0.)
                s_msk_fg = s_msk_fg[:, :, 0:1].transpose(2, 0, 1)
                s_msk_bg = s_msk_bg[:, :, 0:1].transpose(2, 0, 1)
                return s_msk_fg, s_msk_bg
            else:
                q_msk = np.where(img > 0.5, 1., 0.)
                q_msk = q_msk[:, :, 0:1].transpose(2, 0, 1)
                return q_msk
        img = img.transpose(2, 0, 1)        
        return img

    def prepare_query(
        self,
        img_path, 
        n_way=1, # fix ...
        img_size=224,
    ):
        query   = np.zeros([n_way, 3, img_size, img_size], dtype=np.float32)      
        for idx in range(n_way):
            query[idx] = self._preprocess(img_path, img_size=img_size, is_msk=False, is_support=False)
        
        query = torch.from_numpy(query)  
        return query

    def get_feature(self, image_path):
        # Setup image
        self.logger.info('> inference')
        self.logger.info("Read Input Image from : {}".format(image_path))
        q_img = self.prepare_query(image_path, n_way=1, img_size=self.target_img_size)
        q_img = q_img.to(self.device)

        f = self.model.encoder(q_img)
        f = f.view(-1).data.cpu().numpy()
        return f

if __name__ == '__main__':
    curr_file_dir_path = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config",
                        nargs="?",
                        type=str,
                        default=curr_file_dir_path + "/configs/fss.yaml",
                        help="Configuration file to use", )
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = ordered_load(fp, yaml.SafeLoader)

    logdir = cfg['test']['log_dir_path']
    writer = SummaryWriter(log_dir=logdir)
    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)
    shutil.copy(args.config, cfg['test']['result_dir_path'])

    logger = get_logger(logdir)
    logger.info("Let the games begin :)")
    result_dir_path = cfg['test']['result_dir_path']
    feature_path = os.path.join(result_dir_path, '..', 'lung_seg_train_feature.pkl')
    cluster_path = os.path.join(result_dir_path, '..', 'lung_seg_train_cluster.5.pkl')

    ###########################################################################
    # save encoder feature
    ###########################################################################
    # with torch.no_grad():
    #     finder = Finder(cfg, logger)
    #     f_list = []
    #     lung_seg_train_df = pd.read_csv(os.path.join('/data/dawoon/data/xray/CXR_seg_lung', 'chest_seg_lung_train.csv'))
    #     for _, row in tqdm.tqdm(lung_seg_train_df.iterrows()):
    #         row = dict(row)
    #         # img_path = '/data/dawoon/data/xray/CXR_seg_lung/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png/CHNCXR_0597_1.png'
    #         filename = os.path.basename(row['image'])
    #         img_path = os.path.join('/data/dawoon/data/xray/CXR_seg_lung/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png', filename)
    #         f = finder.get_feature(img_path)
    #         f_list.append({'image':filename, 'feature': f})
    #     with open(feature_path, 'wb') as f:
    #         pickle.dump(f_list, f)
    
    ###########################################################################
    # clustering
    ###########################################################################
    with open(feature_path, 'rb') as f:
        f_list = pickle.load(f)

    # new_f_arr = np.array([f['feature']for f in f_list])
    # print('new_f_arr.shape', new_f_arr.shape)

    # cluser = KMeans(n_clusters=5, random_state=2020).fit(new_f_arr)

    # with open(cluster_path, 'wb') as f:
    #     pickle.dump(cluser.cluster_centers_, f)

    ###########################################################################
    # clustering
    ###########################################################################
    with open(cluster_path, 'rb') as f:
        clusters = pickle.load(f)

    best_img_list = []
    from scipy.spatial import distance
    for c in range(clusters.shape[0]):
        c = clusters[c, :]
        
        best_d = 100000
        best_img = None
        for f in f_list:
            d = distance.cosine(c, f['feature'])
            if d < best_d:
                best_d = d
                best_img = f['image']
        
        best_img_list.append(best_img)      
        print(best_img, d)
    
    print('>>>>>> best_img_list')
    print(best_img_list)
    # CHNCXR_0262_0.png 0.19253498315811157
    # CHNCXR_0026_0.png 0.17434006929397583
    # CHNCXR_0154_0.png 0.1299312710762024
    # CHNCXR_0113_0.png 0.10989159345626831
    # CHNCXR_0003_0.png 0.22285813093185425
    # >>>>>> best_img_list
    # ['CHNCXR_0262_0.png', 'CHNCXR_0026_0.png', 'CHNCXR_0154_0.png', 'CHNCXR_0113_0.png', 'CHNCXR_0003_0.png']

