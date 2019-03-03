import torch
from torch.autograd import Variable
import os
from torch import nn
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision import transforms
from model import East
from model import AGD
from loss import *
from data_utils import custom_dset, collate_fn, sampler_for_video_clip, get_proposals
import time
from tensorboardX import SummaryWriter
import config as cfg
from utils.init import *
from utils.util import *
from utils.save import *
from utils.myzip import *
import torch.backends.cudnn as cudnn
from eval import predict
from hmean import compute_hmean
import zipfile
import glob
import warnings
import numpy as np



def train(train_loader, model, AGD_model, criterion1, criterion2, scheduler, optimizer1, optimizer2, epoch):
    start = time.time()
    loss1_ = AverageMeter()
    loss2_ = AverageMeter()
    losst_ = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.train()
    
    for i, (img, score_map, geo_map, training_mask, coord_ids) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if cfg.gpu is not None:
            img, score_map, geo_map, training_mask = img.cuda(), score_map.cuda(), geo_map.cuda(), training_mask.cuda()

        f_score, f_geometry, feature_map = model(img)
        #
        proposals, similarity_mask = get_proposals(score_map, geo_map, coord_ids)
        agd = AGD_model(feature_map, proposals)

        loss1 = criterion1(score_map, f_score, geo_map, f_geometry, training_mask)
        #
        loss2 = criterion2(agd, similarity_mask)

        total_loss = loss1 + loss2 * 0.1

        loss1_.update(loss1.item(), img.size(0))
        loss2_.update(loss2.item(), img.size(0))
        losst_.update(total_loss.item(), img.size(0))

        # backward
        scheduler.step()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        total_loss.backward()
        optimizer1.step()
        optimizer2.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.print_freq == 0:
            print('EAST <==> TRAIN <==> Epoch: [{0}][{1}/{2}] Loss1 {loss1_.val:.4f}({loss1_.avg:.4f}) Loss2 {loss2_.val:.4f}({loss2_.avg:.4f}) Loss_t {losst_.val:.4f}({losst_.avg:.4f})\n'.format(epoch, i, len(train_loader), loss1_=loss1_, loss2_=loss2_,losst_=losst_))
            #print('EAST <==> TRAIN <==> Epoch: [{0}][{1}/{2}] Loss2 {loss.val:.4f} Avg Loss2 {loss.avg:.4f})\n'.format(epoch, i, len(train_loader), loss=loss2_))
            #print('EAST <==> TRAIN <==> Epoch: [{0}][{1}/{2}] Losst_ {loss.val:.4f} Avg Losst_ {loss.avg:.4f})\n'.format(epoch, i, len(train_loader), loss=losst_))
        save_loss_info(loss1_, loss2_, losst_, epoch, i, train_loader)


def main():
    warnings.simplefilter('ignore', np.RankWarning)
    #Model
    video_root_path = os.path.abspath('./dataset/train/')
    video_name_list = sorted([p for p in os.listdir(video_root_path) if p.split('_')[0] == 'Video'])
    #print('video_name_list', video_name_list)
    print('EAST <==> Prepare <==> Network <==> Begin')
    model = East()
    AGD_model = AGD()
    model = nn.DataParallel(model, device_ids=cfg.gpu_ids)
    #AGD_model = nn.DataParallel(AGD_model, device_ids=cfg.gpu_ids)
    model = model.cuda()
    AGD_model = AGD_model.cuda()
    init_weights(model, init_type=cfg.init_type)
    cudnn.benchmark = True

    criterion1 = LossFunc()
    #
    criterion2 = Ass_loss()

    optimizer1 = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    optimizer2 = torch.optim.Adam(AGD_model.parameters(), lr=cfg.lr)
    scheduler = lr_scheduler.StepLR(optimizer1, step_size=10000, gamma=0.94)

    # init or resume
    if cfg.resume and  os.path.isfile(cfg.checkpoint):
        weightpath = os.path.abspath(cfg.checkpoint)
        print("EAST <==> Prepare <==> Loading checkpoint '{}' <==> Begin".format(weightpath))
        checkpoint = torch.load(weightpath)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        #AGD_model.load_state_dict(checkpoint['model2.state_dict'])
        optimizer1.load_state_dict(checkpoint['optimizer'])
        #optimizer2.load_state_dict(checkpoint['optimizer2'])
        print("EAST <==> Prepare <==> Loading checkpoint '{}' <==> Done".format(weightpath))
    else:
        start_epoch = 0
    print('EAST <==> Prepare <==> Network <==> Done')

    for epoch in range(start_epoch+1, cfg.max_epochs):
        for video_name in video_name_list:
            print('EAST <==> epoch:{} <==> Prepare <==> DataLoader <==>{} Begin'.format(epoch, video_name))
            trainset = custom_dset(os.path.join(video_root_path, video_name))
            #sampler = sampler_for_video_clip(len(trainset))
            train_loader = DataLoader(trainset, batch_size=cfg.train_batch_size_per_gpu*cfg.gpu,
                shuffle=False, collate_fn=collate_fn , num_workers=cfg.num_workers, drop_last=True)
            print('EAST <==> Prepare <==> Batch_size:{} <==> Begin'.format(cfg.train_batch_size_per_gpu*cfg.gpu))
            print('EAST <==> epoch:{} <==> Prepare <==> DataLoader <==>{} Done'.format(epoch, video_name))

            train(train_loader, model, AGD_model, criterion1, criterion2, scheduler, optimizer1, optimizer2, epoch)
            '''
            for i, (img, score_map, geo_map, training_mask, coord_ids) in enumerate(train_loader):
                print('i{} img.shape:{} geo_map.shape{} training_mask.shape{} coord_ids.len{}'.format(i, score_map.shape, geo_map.shape, training_mask.shape, len(coord_ids)))
            '''

        if epoch % cfg.eval_iteration == 0:
            state = {
                    'epoch'             : epoch,
                    'model1.state_dict' : model.state_dict(),
                    'model2.state_dict' : AGD_model.state_dict(),
                    'optimizer1'        : optimizer1.state_dict(),
                    'optimizer2'        : optimizer2.state_dict()
                    }
            save_checkpoint(state, epoch)
                



if __name__ == "__main__":
    main()
