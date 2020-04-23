import os
import shutil
import datetime
import argparse
import time
import tqdm
import yaml
import numpy as np
import pandas as pd

from tensorboardX import SummaryWriter
import torch
import cv2
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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


def compute_iou(gt_qmask, pr_qmask):
    gt_qmask = gt_qmask.reshape(-1)
    pr_qmask = pr_qmask.reshape(-1)
    tn, fp, fn, tp = confusion_matrix(gt_qmask, pr_qmask).ravel()
    iou = tp / ( tp + fp + fn)

    return iou

def compute_dice(gt_qmask, pr_qmask):
    gt_qmask = gt_qmask.reshape(-1)
    pr_qmask = pr_qmask.reshape(-1)
    tn, fp, fn, tp = confusion_matrix(gt_qmask, pr_qmask).ravel()
    dice = 2 * tp / ( 2 * tp + fp + fn)

    return dice


def compute_acc(gt_qmask, pr_qmask):
    """ pixel accuacy
    """
    gt_qmask = gt_qmask.reshape(-1)
    pr_qmask = pr_qmask.reshape(-1)
    acc = accuracy_score(gt_qmask, pr_qmask)
    return acc
    

def compute_f1(gt_qmask, pr_qmask):
    """ pixel accuacy
    """
    gt_qmask = gt_qmask.reshape(-1)
    pr_qmask = pr_qmask.reshape(-1)
    acc = f1_score(gt_qmask, pr_qmask)
    return acc

class Infer(object):    
    def __init__(self, cfg, logger=None):
        self.cfg = cfg
        self.logger = logger
        self.device = setup_device(cfg.get('gpus', '0'))
        # self.device = 'cpu'
        self.model = None
        self.load_model()
        self.iou = 0
        self.target_img_size = self.cfg['augmentations']['valid_augmentations']['np_scale']
        self.support, self.smasks_fg, self.smasks_bg = self.prepare_support(
            self.cfg['test']['support_list'], 
            n_way=1, 
            k_shot=5,
            img_size=self.target_img_size,
        )
        self.support = self.support.to(self.device)
        self.smasks_fg = self.smasks_fg.to(self.device)
        self.smasks_bg = self.smasks_bg.to(self.device)

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

    def prepare_support(
        self,
        support_info, 
        n_way=1, # fix ...
        k_shot=5,
        img_size=224,
    ):
        support    = np.zeros([n_way, k_shot, 3, img_size, img_size], dtype=np.float32)
        smasks_fg  = np.zeros([n_way, k_shot, 1, img_size, img_size], dtype=np.float32)
        smasks_bg  = np.zeros([n_way, k_shot, 1, img_size, img_size], dtype=np.float32)

        for idx in range(n_way):
            for idy in range(k_shot):
                for s in support_info:
                    support[idx, idy] = self._preprocess(s['image'], img_size=img_size, is_msk=False, is_support=True)
                    s_msk_fg, s_msk_bg = self._preprocess(s['mask'], img_size=img_size, is_msk=True, is_support=True)
                    smasks_fg[idx, idy] = s_msk_fg
                    smasks_bg[idx, idy] = s_msk_bg
        
        support = torch.from_numpy(support)  
        smasks_fg = torch.from_numpy(smasks_fg)  
        smasks_bg = torch.from_numpy(smasks_bg)  
        return support, smasks_fg, smasks_bg


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

    def prepare_query_mask(
        self,
        img_path, 
        n_way=1, # fix ...
        img_size=224,
    ):
        qmask   = np.zeros([n_way, 1, img_size, img_size], dtype=np.float32)      
        for idx in range(n_way):
            qmask[idx] = self._preprocess(img_path, img_size=img_size, is_msk=True, is_support=False)
        return qmask    

    def inference(self, image_path, result_dir_path, mask_path=''):
        # Setup image
        self.logger.info('> inference')
        self.logger.info("Read Input Image from : {}".format(image_path))
        self.origin_q_img = cv2.imread(img_path)
        q_img = self.prepare_query(image_path, n_way=1, img_size=self.target_img_size)
        q_img = q_img.to(self.device)
        query_pred, _ = self.model(self.support, self.smasks_fg, self.smasks_bg, q_img)
        pred_msk = query_pred.argmax(dim=1)[0].data.cpu().numpy()

        _pred_msk = pred_msk.reshape(self.target_img_size, self.target_img_size, 1)
        _zero_padding = np.zeros(_pred_msk.shape)
        pred_msk_img = np.concatenate((
            _pred_msk, 
            _zero_padding, 
            _zero_padding), axis=2) * 255 # blue

        pred_msk_img = cv2.resize(pred_msk_img, (self.origin_q_img.shape[1], self.origin_q_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        save_img_list = [
            self.origin_q_img,
            self.origin_q_img * 0.7 + pred_msk_img * 0.3,
        ]
        
        if mask_path != '':
            origin_q_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            _origin_q_mask = origin_q_mask.reshape(origin_q_mask.shape[0], origin_q_mask.shape[1], 1)
            _zero_padding = np.zeros(_origin_q_mask.shape)
            origin_q_mask = np.concatenate((
                _zero_padding, 
                _origin_q_mask, 
                _zero_padding), axis=2) # green
            save_img_list.append(self.origin_q_img * 0.7 + origin_q_mask * 0.3)
            save_img_list.append(pred_msk_img  + origin_q_mask)

            q_msk = self.prepare_query_mask(mask_path, n_way=1, img_size=self.target_img_size)
            # q_msk = torch.cat([query_label.long().to(device) for query_label in q_msk[0]], dim=0)
            q_msk = q_msk.reshape((self.target_img_size, self.target_img_size))
            self.iou = compute_iou(q_msk, pred_msk)
            self.dice = compute_dice(q_msk, pred_msk)
            self.acc = compute_acc(q_msk, pred_msk)
            self.f1 = compute_f1(q_msk, pred_msk)
            

        img = np.concatenate(save_img_list, axis=1)
        img = img.astype(np.uint8)

        fname, ext = os.path.splitext(os.path.basename(image_path))
        result_dir_path = os.path.join(result_dir_path, fname + \
                            '.iou(' +str(np.round(self.iou, 4)) +')'+ \
                            '.dice(' +str(np.round(self.dice, 4)) +')'+ \
                            '.acc(' +str(np.round(self.acc, 4)) +')'+ \
                            '.f1(' +str(np.round(self.f1, 4)) +')'+ \
                            ext)
        print(result_dir_path)
        img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
        cv2.imwrite(result_dir_path, img)


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

    with torch.no_grad():
        infer = Infer(cfg, logger)
        result_dir_path = cfg['test']['result_dir_path']
        
        lung_seg_test_df = pd.read_csv(os.path.join('/data/dawoon/data/xray/CXR_seg_lung', 'chest_seg_lung_test.csv'))
        report_list = []
        for _, row in tqdm.tqdm(lung_seg_test_df.iterrows()):
            row = dict(row)
            image = os.path.basename(row['image'])
            mask = os.path.basename(row['mask'])
            img_path = os.path.join('/data/dawoon/data/xray/CXR_seg_lung/ChinaSet_AllFiles/ChinaSet_AllFiles/CXR_png', image)
            msk_path = os.path.join('/data/dawoon/data/xray/CXR_seg_lung/shcxr-lung-mask/mask/mask', mask)
            infer.inference(img_path, result_dir_path, mask_path=msk_path)
            report_list.append({
                'image': image,
                'mask': mask,
                'iou': infer.iou,
                'dice': infer.dice,
                'acc': infer.acc,
                'f1': infer.f1
            })
        
        report_df = pd.DataFrame(report_list)
        report_df.to_csv(os.path.join(result_dir_path, '..', 'lung_seg_report.csv'), index=False)
        print('>>> mean iou', np.mean(report_df['iou'].values))
        print('>>> mean dice', np.mean(report_df['dice'].values))
        print('>>> mean acc', np.mean(report_df['acc'].values))
        print('>>> mean f1', np.mean(report_df['f1'].values))
