import os
import shutil
import datetime
import argparse
import time
import tqdm
import numpy as np
import yaml
from tensorboardX import SummaryWriter
import torch

from sklearn.metrics import f1_score, accuracy_score

from core.utils.utils import setup_device, setup_seeds

from core.utils.misc import ordered_load
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
    intersection = np.sum(gt_qmask * pr_qmask)
    union = gt_qmask + pr_qmask
    union = np.sum(np.where(union>= 1, 1. , 0.))
    miou = intersection/union
    # pr_qmask = np.where(pr_qmask> 0.5, 1. , 0.)
    # for idx in range(pr_qmask.shape[0]):
    #     notTrue = 1 -  gt_qmask[idx]
    #     union = np.sum(gt_qmask[idx] + (notTrue * pr_qmask[idx]))
    #     intersection = np.sum(gt_qmask[idx] * pr_qmask[idx])
    #     ious += (intersection / union)
    # miou = (ious / pr_qmask.shape[0])
    return miou

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


def train(cfg, writer, logger):
    # Setup seeds
    seed = cfg['data'].get('seed', 1336)
    setup_seeds(seed)

    # Setup device
    device = setup_device(cfg.get('gpus', '0'))

    # Setup Augmentations
    train_aug = get_composed_augmentations(cfg["augmentations"].get("train_augmentations", None))
    valid_aug = get_composed_augmentations(cfg["augmentations"].get("valid_augmentations", None))

    # Setup Dataloader
    dataset_cls = get_dataset(cfg["data"]["dataset"])
    train_label_path_list = cfg["data"]["train_label_path_list"]
    valid_label_path = cfg["data"]["valid_label_path"]
    num_classes = cfg["data"]["num_classes"]
    batch_size = cfg["train"]["batch_size"]
    num_workers = cfg["train"]["n_workers"]
    data_root = cfg["data"]["data_root"]
    x_key = cfg["data"]["x_key"]
    # [train_class, test_class] = split_dataset_by_csv(train_label_path_list, valid_label_path, x_key=x_key)

    train_loader, valid_loader = setup_dataloader(
        dataset_cls, train_label_path_list, valid_label_path,
        x_key=x_key, data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers, train_aug=train_aug, valid_aug=valid_aug)

    logger.info('len(train_loader): {}'.format(len(train_loader)))
    logger.info('len(valid_loader): {}'.format(len(valid_loader)))

    # Setup Model
    model = get_model(cfg["model"], num_classes).to(device)
    # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.train()

    logger.info('>> Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # Setup optimizer, lr_scheduler
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["train"]["optimizer"].items() if k != "name"}
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))
    scheduler_name = cfg["train"]["lr_schedule"]["name"]
    scheduler = get_scheduler(optimizer, cfg["train"]["lr_schedule"])

    # Setup loss function
    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))

    start_epoch = 1
    if cfg["train"]["resume"] is not None:
        if os.path.isfile(cfg["train"]["resume"]):
            logger.info("Loading model and optimizer from checkpoint '{}'".format(cfg["train"]["resume"]))
            checkpoint = torch.load(cfg["train"]["resume"])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"] + 1
            logger.info("Loaded checkpoint '{}' (epoch {})".format(cfg["train"]["resume"], checkpoint["epoch"]))
        else:
            logger.info("No checkpoint found at '{}'".format(cfg["train"]["resume"]))

    ####################################################################################################################
    #  epoch
    ####################################################################################################################
    curr_epoch = start_epoch
    valid_step_list = list(np.linspace(0, len(train_loader), num=cfg['train']['n_valid_per_epoch']+1, endpoint=True))[1:]
    valid_step_list = [int(step) for step in valid_step_list]
    print('valid_step_list', valid_step_list)
    while curr_epoch <= cfg["train"]["n_epoch"]:
        start_ts = time.time()
        train_q_loss_meter = AverageMeter()
        train_a_loss_meter = AverageMeter()
        valid_q_loss_meter = AverageMeter()
        valid_iou_meter = AverageMeter()
        valid_acc_meter = AverageMeter()
        valid_f1_meter = AverageMeter()
        ################################################################################################################
        #  train
        ################################################################################################################
        for train_i, (train_support, train_smasks_fg, train_smasks_bg, train_query, train_qmask, _) in enumerate(tqdm.tqdm(train_loader)):
            model.train()
            # print('train_support', train_support.shape)
            # print('train_smasks_fg', train_smasks_fg.shape)
            # print('train_query', train_query.shape)
            # print('train_qmask', train_qmask.shape)
            # train_support torch.Size([1, 1, 5, 3, 224, 224])
            # train_smasks_fg torch.Size([1, 1, 5, 1, 224, 224])
            # train_query torch.Size([1, 1, 3, 224, 224])
            # train_qmask torch.Size([1, 1, 1, 224, 224])

            # Prepare input (batch 차원 제거...)
            support_images = train_support[0].to(device)
            support_fg_mask = train_smasks_fg[0].float().to(device)
            support_bg_mask = train_smasks_bg[0].float().to(device)

            query_images = train_query[0].to(device)
            query_labels = torch.cat([query_label.long().to(device) for query_label in train_qmask[0]], dim=0)

            # Forward and Backward
            optimizer.zero_grad()
            query_pred, align_loss = model(support_images, support_fg_mask, support_bg_mask, query_images)
            query_loss = loss_fn(query_pred, query_labels)
            loss = query_loss + align_loss * 1 # _config['align_loss_scaler']
            loss.backward()
            optimizer.step()
            tensor_type_str = "<class 'torch.Tensor'>"
            if (str(type(query_loss)) == tensor_type_str) and (str(type(align_loss)) == tensor_type_str):
                train_q_loss_meter.update(query_loss.data.cpu().numpy())
                train_a_loss_meter.update(align_loss.data.cpu().numpy()) # AttributeError: 'float' object has no attribute 'data'
            else:
                print('>>> type(loss) is not tensor.')
                print('>>> query_loss: ', query_loss)
                print('>>> align_loss: ', align_loss)

            n_step = train_i + 1
            n_step_global = int((curr_epoch - 1) * len(train_loader) + n_step)
            ############################################################################################################
            #  validation
            ############################################################################################################
            if n_step in valid_step_list:
                # gt_all = torch.FloatTensor().to(device)
                # pred_all = torch.FloatTensor().to(device)
                # model.eval()
                with torch.no_grad():
                    for valid_i, (valid_support, valid_smasks_fg, valid_smasks_bg, valid_query, valid_qmask, q_img_path) in enumerate(valid_loader):
                        # Prepare input (batch 차원 제거...)
                        _support_images = valid_support[0].to(device)
                        _support_fg_mask = valid_smasks_fg[0].float().to(device)
                        _support_bg_mask = valid_smasks_bg[0].float().to(device)

                        _query_images = valid_query[0].to(device)
                        _query_labels = torch.cat([query_label.long().to(device) for query_label in valid_qmask[0]], dim=0)

                        _query_pred, _ = model(_support_images, _support_fg_mask, _support_bg_mask, _query_images)

                        _query_loss = loss_fn(_query_pred, _query_labels)


                        tensor_type_str = "<class 'torch.Tensor'>"
                        if str(type(_query_loss)) == tensor_type_str:
                            valid_q_loss_meter.update(_query_loss.data.cpu().numpy())
                        else:
                            print('>>> type(loss) is not tensor.')
                            print('>>> valid query_loss: ', _query_loss)

                        _query_pred = _query_pred.argmax(dim=1)[0].data.cpu().numpy()
                        _query_labels =  _query_labels[0].data.cpu().numpy()
                        # print('query_pred.shape', query_pred.shape)
                        # print('query_labels.shape', query_labels.shape)
                        # query_pred.shape (224, 224)
                        # query_labels.shape (224, 224)
                        # index, count = np.unique(query_pred, return_counts=True)
                        # print('count query_pred', index, count)
                        # index, count = np.unique(query_labels, return_counts=True)
                        # print('count query_labels', index, count)
                        # pred_y = torch.sigmoid(pred_y)
                        # pred_y = torch.round(pred_y)
                            
                        iou = compute_iou(_query_labels, _query_pred)
                        acc = compute_acc(_query_labels, _query_pred)
                        f1  = compute_f1( _query_labels, _query_pred)
                        valid_iou_meter.update(iou)
                        valid_acc_meter.update(acc)
                        valid_f1_meter.update(f1)

                train_loss = np.round(train_q_loss_meter.avg, 4)
                train_loss_a = np.round(train_a_loss_meter.avg, 4)
                valid_loss = np.round(valid_q_loss_meter.avg, 4)
                valid_iou = np.round(valid_iou_meter.avg, 4)
                valid_acc = np.round(valid_acc_meter.avg, 4)
                valid_f1 = np.round(valid_f1_meter.avg, 4)
                train_q_loss_meter.reset()
                train_a_loss_meter.reset()
                valid_q_loss_meter.reset()
                valid_iou_meter.reset()
                valid_acc_meter.reset()
                valid_f1_meter.reset()
                currunt_lr = 0
                for param_group in optimizer.param_groups:
                    currunt_lr = param_group['lr']
                
                logger.info(f'>> Epoch[{int(curr_epoch)}/{int(cfg["train"]["n_epoch"])}]')
                logger.info(f'>> {datetime.datetime.now()} Train_Loss(q): {train_loss} Train_Loss(a): {train_loss_a}')
                logger.info(f'>> {datetime.datetime.now()} Validation_Loss: {valid_loss} IoU:{valid_iou} acc:{valid_acc} f1:{valid_f1} Currunt LR: {currunt_lr}')
                state = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": curr_epoch,
                }
                save_name = cfg["model"]["arch"] + '_' + cfg["data"]["dataset"] + \
                            '.iou(' +  str(valid_iou) + ').' + \ 
                            '.acc(' +  str(valid_acc) + ').' + \ 
                            '.f1(' +  str(valid_f1) + ').' + \ 
                            '.epoch(' +  str(curr_epoch) + ').' + \ 
                            ".pth.tar" 
                # metrics.save_model_state(state, save_path=cfg["train"]["save_dir_path"], save_name=save_name)
                torch.save(state, os.path.join(cfg["train"]["save_dir_path"], save_name))

                writer.add_scalar("loss/train_loss", train_loss, n_step_global)
                writer.add_scalar("loss/tarin_loss_a", train_loss_a, n_step_global)
                writer.add_scalar("loss/valid_loss", valid_loss, n_step_global)
                writer.add_scalar("loss/valid_iou", valid_iou, n_step_global)
                writer.add_scalar("loss/valid_acc", valid_acc, n_step_global)
                writer.add_scalar("loss/valid_f1", valid_f1, n_step_global)
                writer.add_scalar("learning_rate", currunt_lr, n_step_global)
                start_ts = time.time()

        curr_epoch += 1
        scheduler.step()


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

    logdir = cfg['train']['log_dir_path']
    writer = SummaryWriter(log_dir=logdir)
    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)
    shutil.copy(args.config, cfg['train']['save_dir_path'])

    logger = get_logger(logdir)
    logger.info("Let the games begin :)")
    train(cfg, writer, logger)