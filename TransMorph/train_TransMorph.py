from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader

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
    cuda_idx = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_idx  # Choose GPU
    batch_size = 1
    num_workers = 4
    atlas_path = None  # when its none, we are training brain2brain (and not brain2atlas)
    # atlas_path = '/raid/data/users/adarc/registration/IXI/IXI_data/new_atlas_subject_6_from_test.pkl'
    train_dir = '/raid/data/users/adarc/registration/IXI/IXI_data/Train/'
    # train_dir = r"D:\Datasets\Learning2Reg\L2R_2021_Task3_test/"
    # val_dir = r"D:\Datasets\Learning2Reg\L2R_2021_Task3_test/"
    val_dir = '/raid/data/users/adarc/registration/IXI/IXI_data/Val/'
    weights = [1, 0.02] # loss weights
    mi_loss = lambda x, y: losses.diff_mutual_information(x, y)
    mi_from_other_repo = MutualInformationFromOtherRepo(num_bins=256, sigma=0.1, normalize=True).cuda()
    mi_loss = lambda x, y: 1.3 + 1 - mi_from_other_repo(x, y)
    L2_loss = nn.MSELoss()
    grad3d_loss = losses.Grad3d(penalty='l2')
    criterions = [mi_loss, grad3d_loss]
    penalty_lambda = 3
    # process_name = f'DEBUG__delete_this'
    process_name = f'50%faster_MI_from_other_repo_plus1.3__3_penalty_atlas2atlas_IXI_cuda{cuda_idx}'
    if not os.path.exists('experiments/'+process_name):
        os.makedirs('experiments/'+process_name)
    if not os.path.exists('logs/'+process_name):
        os.makedirs('logs/'+process_name)
    sys.stdout = Logger('logs/'+process_name)
    lr = 0.0001 # learning rate

    epoch_start = 0
    max_epoch = 500 #max traning epoch
    # pretrained_model_path = '/raid/data/users/adarc/registration/forked-remote/experiments/bs1_onlydataset_change_TransMorph_mse_1_diffusion_0.02/dsc0.694.pth.tar'
    pretrained_model_path = None

    '''
    Initialize model
    '''
    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)
    model.cuda()

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(config.img_size, 'nearest')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.cuda()

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
    train_set = datasets.JHUBrainDataset(glob.glob(train_dir + '*.pkl'), transforms=train_composed, atlas_path=atlas_path)
    val_set = datasets.JHUBrainDataset(glob.glob(val_dir + '*.pkl'), transforms=val_composed, atlas_path=atlas_path)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)


    best_dsc = 0
    writer = SummaryWriter(log_dir='logs/'+process_name)
    for epoch in range(epoch_start, max_epoch):
        print(f'starting epoch {epoch}')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        model.train()
        for train_batch_idx, data in enumerate(train_loader):
            # if train_batch_idx>2:
            #     break
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)

            data = [t.cuda() for t in data]
            x, x_seg, y, y_seg = data
            # y_t2 = _get_t2_approx(x, y)

            # x_in = torch.cat((x, y_t2), dim=1)
            x_in = torch.cat((x,y), dim=1)

            output = model(x_in)
            # ~~~~ Old loss ~~~~
            # loss = 0
            # loss_vals = []
            # for n, loss_function in enumerate(criterions):
            #     curr_loss = loss_function(output[n], y) * weights[n]
            #     loss_vals.append(curr_loss)
            #     loss += curr_loss
            # loss_all.update(loss.item(), y.numel())
            with torch.no_grad():
                input_mi = mi_loss(x, y)
                output_l2 = L2_loss(output[0], y)  # l2 compare to original t1
            output_mi = mi_loss(output[0], y)
            loss = output_mi + penalty_lambda * grad3d_loss(output[1], y)   # isntead of 0.02

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss/train_batch', loss.item(), train_batch_writer_step)
            writer.add_scalar('Loss/train_batch_input_mi', input_mi.item(), train_batch_writer_step)
            writer.add_scalar('Loss/train_batch_output_mi', output_mi.item(), train_batch_writer_step)
            writer.add_scalar('Loss/train_batch_output_l2', output_l2.item(), train_batch_writer_step)

            with torch.no_grad():
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                dsc = utils.dice_val(def_out.long(), y_seg.long(), 46)
            writer.add_scalar('DSC/train_batch', dsc.item(), train_batch_writer_step)
            train_batch_writer_step+=1

            del x_in
            del output
            del loss
            # flip fixed and moving images
            # x_t2 = _get_t2_approx(y, x)
            # x_in = torch.cat((y, x_t2), dim=1)
            x_in = torch.cat((y, x), dim=1)
            output = model(x_in)
            # ~~~~ Old loss ~~~~
            # loss = 0
            # for n, loss_function in enumerate(criterions):
            #     curr_loss = loss_function(output[n], x) * weights[n]
            #     loss_vals[n] += curr_loss
            #     loss += curr_loss
            # loss_all.update(loss.item(), y.numel())

            # loss = L2_loss(output[0], x)
            # loss = L2_loss(output[0], x_t2) + penalty_lambda * grad3d_loss(output[1], x_t2)  # isntead of 0.02

            output_mi = mi_loss(output[0], x)
            loss = output_mi + penalty_lambda * grad3d_loss(output[1], x)  # isntead of 0.02



            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch}, Iter {train_batch_idx}/{len(train_loader)}: Loss: {loss.item():.3f}, DSC: {dsc.item():.3f}')

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
                x_in = torch.cat((x, y), dim=1)
                grid_img = mk_grid_img(8, 1, config.img_size)
                output = model(x_in)
                loss = 0
                for n, loss_function in enumerate(criterions):
                    curr_loss = loss_function(output[n], y) * weights[n]
                    loss += curr_loss
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                def_grid = reg_model_bilin([grid_img.float(), output[1].cuda()])
                dsc = utils.dice_val(def_out.long(), y_seg.long(), 46)
                writer.add_scalar('Loss/val_batch', loss.item(), val_batch_writer_step)
                writer.add_scalar('DSC/val_batch', dsc.item(), val_batch_writer_step)
                val_batch_writer_step += 1
                eval_dsc.update(dsc.item(), x.size(0))
                print(eval_dsc.avg)
        if eval_dsc.avg > best_dsc:
            best_dsc = eval_dsc.avg
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_dsc': best_dsc,
                'optimizer': optimizer.state_dict(),
            }, save_dir=f'experiments/{process_name}', filename='best_dsc_checkpoint')
        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
        plt.switch_backend('agg')
        pred_fig = comput_fig(def_out)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x_seg)
        tar_fig = comput_fig(y_seg)
        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure('input', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('ground truth', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch)
        plt.close(pred_fig)
        loss_all.reset()
    writer.close()


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