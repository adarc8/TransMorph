from PIL.Image import fromarray
from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader

from models.unet_from_github.unet import UNet
from deep_hist_repo.MI_from_other_repo import MutualInformationFromOtherRepo
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def main():
    #get num_workers from argpasrse:
    num_workers = _get_from_argparse()
    working_remotely = os.getcwd().split('/')[1] == 'raid'
    cuda_idx = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_idx  # Choose GPU
    batch_size = 1
    atlas_path = None  # when its none, we are training brain2brain (and not brain2atlas)
    if working_remotely:
        # IXI or OASIS
        # data_root = '/raid/data/users/adarc/registration/data/OASIS/images_labels_Tr_pkls'
        data_root = '/raid/data/users/adarc/registration/data/IXI/IXI_data'
    else:
        data_root = r"D:\Datasets\Learning2Reg\OASIS_2022\images_labels_Tr_pkls"

    # atlas_path = os.path.join(data_root, 'new_atlas_subject_6_from_test.pkl')
    train_dir = os.path.join(data_root, 'Train')
    val_dir = os.path.join(data_root, 'Val')

    weights = [1, 0.02] # loss weights
    # mi_loss = lambda x, y: (losses.diff_mutual_information(x, y)).mean()
    mi_loss = lambda x, y: (losses.diff_mutual_information(x, y, n_channels_avg=0)).mean()
    # mi_from_other_repo = MutualInformationFromOtherRepo(num_bins=256, sigma=0.1, normalize=True).cuda()
    # mi_loss = lambda x, y: 1 - mi_from_other_repo(x, y)
    L2_loss = nn.MSELoss()
    grad3d_loss = losses.Grad3d(penalty='l2')
    cross_entropy_loss = nn.CrossEntropyLoss()
    criterions = [mi_loss, grad3d_loss]
    penalty_lambda = 3
    n_classes = 2
    # process_name = f'DEBUG__delete_this'
    supervised = False
    process_name = f'{n_classes=}_{supervised=}_MI_IXI_cuda{cuda_idx}'
    short_dataset = True if "short" in process_name else False
    if not os.path.exists('experiments/'+process_name):
        os.makedirs('experiments/'+process_name)
    if not os.path.exists('logs/'+process_name):
        os.makedirs('logs/'+process_name)
    sys.stdout = Logger('logs/'+process_name)
    lr = 1e-4  # learning rate 0.001

    epoch_start = 0
    max_epoch = 50000 #max traning epoch
    # pretrained_model_path = '/raid/data/users/adarc/registration/forked-remote/experiments/bs1_onlydataset_change_TransMorph_mse_1_diffusion_0.02/dsc0.694.pth.tar'
    pretrained_model_path = None

    '''
    Initialize model
    '''
    config = CONFIGS_TM['TransMorph']
    model = UNet(n_outputs=n_classes)
    # model = TransMorph.TransMorph(config)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    # reg_model = utils.register_model(config.img_size, 'nearest')
    # reg_model.cuda()
    # reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    # reg_model_bilin.cuda()

    # If continue from previous training
    if pretrained_model_path is not None:
        epoch_start = 394
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(pretrained_model_path)['state_dict']
        print(f'Loading Pretrained from: {pretrained_model_path}')
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_batch_writer_step = 0
    val_batch_writer_step = 0
    train_composed = transforms.Compose([
        trans.Seg_norm(),  # rearrange segmentation label to 1 to 46
        trans.RandomFlip(0),  # flip along x axis
        trans.NumpyType((np.float32, np.int16, np.float32, np.int16)),  # convert to float32
    ])
    val_composed = transforms.Compose([
        trans.Seg_norm(),  # rearrange segmentation label to 1 to 46
        trans.NumpyType((np.float32, np.int16, np.float32, np.int16)),
    ])
    train_set = datasets.JHUBrainDataset(glob.glob(os.path.join(train_dir, '*.pkl')), transforms=train_composed, atlas_path=atlas_path, n_classes=n_classes, short_dataset=short_dataset)
    val_set = datasets.JHUBrainDataset(glob.glob(os.path.join(val_dir, '*.pkl')), transforms=val_composed, atlas_path=atlas_path, n_classes=n_classes, short_dataset=short_dataset)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)


    best_dsc = 0
    writer = SummaryWriter(log_dir='logs/'+process_name)
    for epoch in range(epoch_start, max_epoch):
        print(f'starting epoch {epoch} of {process_name=}')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        model.train()
        for train_batch_idx, data in enumerate(train_loader):
            # if train_batch_idx>=0:
            #     break
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)

            data = [t.cuda() for t in data]
            x, x_seg, y, y_seg = data
            # y_t2 = _get_t2_approx(x, y)

            # concat on batch dim
            x_in = torch.cat((x, y), dim=0)
            seg_gt = torch.cat((x_seg, y_seg), dim=0)[:, 0]  # remove the channel dim (only 1)
            # x_in = torch.cat((x, y_t2), dim=1)
            output = model(x_in)
            seg_gt_seg_dim = _add_seg_channel_dim(output, seg_gt)
            if supervised:
                # seg_gt is (BS, H, W, D). Output is (BS, C, H, W, D)
                # so we need to add a channel dim to seg_gt
                # now lets use cross entropy loss
                ce_output_seg = cross_entropy_loss(output, seg_gt_seg_dim)
                with torch.no_grad():
                    mi_in_output = mi_loss(x_in, output)
                loss = ce_output_seg
            else:
                mi_in_output = mi_loss(x_in, output)
                with torch.no_grad():
                    ce_output_seg = cross_entropy_loss(output, seg_gt_seg_dim)
                loss = mi_in_output
            # check if loss is nan
            if torch.isnan(loss):
                print(f'{train_batch_idx=}: Loss is nan! skipping this batch')
                continue
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                output_seg = torch.argmax(output, dim=1)
                # [0,1,2,3] -> [0, 0.33, 0.66, 1]
                output_seg = output_seg.float() / (n_classes-1)
                seg_gt = seg_gt.float() / (n_classes-1)
                seg_gt = torch.clamp(seg_gt, 0, 1)  # some how gt has couple values out of [0,1]
                seg_mi = mi_loss(output_seg, seg_gt)

            # compute gradient and do SGD step


            writer.add_scalar('Loss/train_batch', loss.item(), train_batch_writer_step)
            writer.add_scalar('Loss/train_seg_mi', seg_mi.item(), train_batch_writer_step)
            writer.add_scalar('Loss/train_mi_in_output', mi_in_output.item(), train_batch_writer_step)
            writer.add_scalar('Loss/train_ce_output_seg', ce_output_seg.item(), train_batch_writer_step)
            train_batch_writer_step += 1

            print(f'Epoch {epoch}, Iter {train_batch_idx}/{len(train_loader)}: Loss: {loss.item():.3f}')

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))
        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        model.eval()
        with torch.no_grad():
            for val_batch_idx, data in enumerate(val_loader):
                # if val_batch_idx > 2:
                #     break
                data = [t.cuda() for t in data]
                x, x_seg, y, y_seg = data
                # y_t2 = _get_t2_approx(x, y)
                # x_in = torch.cat((x, y_t2), dim=1)
                x_in = torch.cat((x, y), dim=0)
                seg_gt = torch.cat((x_seg, y_seg), dim=0)[:, 0]  # remove the channel dim (only 1)
                output = model(x_in)
                seg_gt_seg_dim = _add_seg_channel_dim(output, seg_gt)
                ce_output_seg = cross_entropy_loss(output, seg_gt_seg_dim)
                mi_in_output = mi_loss(x_in, output)
                loss = ce_output_seg if supervised else mi_in_output
                if val_batch_idx == 0:
                    # save example to disk
                    output_to_save = torch.argmax(output[0], dim=0)
                    output_to_save_idx100 = output_to_save[:, 20].cpu().numpy()
                    output_to_save_idx100 = (255*(output_to_save_idx100/(n_classes-1))).astype(np.uint8)
                    output_to_save_idx100 = fromarray(output_to_save_idx100)
                    output_to_save_idx100.save(f'experiments/{process_name}/output_{epoch=}.png')

                with torch.no_grad():
                    output_seg = torch.argmax(output, dim=1)
                    # [0,1,2,3] -> [0, 0.33, 0.66, 1]
                    output_seg = output_seg.float() / (n_classes-1)
                    seg_gt = seg_gt.float() / (n_classes-1)
                    seg_gt = torch.clamp(seg_gt, 0, 1)  # some how gt has couple values out of [0,1]
                    seg_mi = mi_loss(output_seg, seg_gt)

                writer.add_scalar('Loss/val_batch', loss.item(), val_batch_writer_step)
                writer.add_scalar('Loss/val_seg_mi', seg_mi.item(), val_batch_writer_step)
                writer.add_scalar('Loss/val_mi_in_output', mi_in_output.item(), val_batch_writer_step)
                writer.add_scalar('Loss/val_ce_output_seg', ce_output_seg.item(), val_batch_writer_step)
                val_batch_writer_step += 1
                # print(eval_dsc.avg)
    writer.close()


def _add_seg_channel_dim(output, seg_gt):
    seg_gt2 = torch.zeros(output.shape).cuda()
    for channel_idx in range(output.shape[1]):
        seg_gt2[:, channel_idx] = seg_gt == channel_idx
    return seg_gt2


def _get_from_argparse():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    num_workers = args.num_workers
    return num_workers


def _get_t2_approx(x, y):
    y_t2 = 1 - y  # scans histogram is mostly in [0,0.6], so t2 is in [0.4, 1]
    # so we shift it by this "0.4" which we calc by the 3% percentile
    perc3 = torch.kthvalue(y_t2[y_t2 > 0], int(y_t2[y_t2 > 0].numel() * 0.03)).values
    y_t2 -= perc3
    y_t2[y_t2 < 0] = 0
    background_mask = x > 0.05
    y_t2 = y_t2 * background_mask
    return y_t2


def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12,12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig

def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, os.path.join(save_dir, filename))

if __name__ == '__main__':
    main()