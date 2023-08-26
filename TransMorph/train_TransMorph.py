from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
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

def main(running_on_dgx, num_workers=2, batch_size=4, half_precision=True):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    if running_on_dgx:
        database_dir = Path('/raid/data/users/adarc/registration/IXI/IXI_data')
    else:
        database_dir = Path(r'D:\Datasets\IXI\ixi_from_transmorph\IXI_data')  #

    train_dir = database_dir / 'Train'
    val_dir = database_dir / 'Val'
    train_writer_step = 0
    val_writer_step = 0
    weights = [1, 0.02]  # loss weights
    process_name = f'HP3_TransMorph_mse_{weights[0]}_diffusion_{weights[1]}'
    experiment_dir = f'output/experiments/{process_name}'
    logs_dir = f'output/logs/{process_name}'

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    sys.stdout = Logger(logs_dir)
    lr = 0.0001  # learning rate
    epoch_start = 0
    max_epoch = 500  #max traning epoch
    cont_training = False  #if continue training

    '''
    Initialize model
    '''
    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)
    model.to(device)

    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(device, config.img_size, 'nearest')  # todo remove inputing device (no use anymore)
    reg_model.to(device)
    reg_model_bilin = utils.register_model(device, config.img_size, 'bilinear')
    reg_model_bilin.to(device)

    '''
    If continue from previous training
    '''
    if cont_training:
        epoch_start = 394
        model_dir = experiment_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch,0.9),8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-2])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-2]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose([
        trans.RandomFlip(0),
        trans.NumpyType((np.float32, np.float32))])
    # rearrange segmentation label to 1 to 46
    val_composed = transforms.Compose([
        trans.Seg_norm(),
        trans.NumpyType((np.float32, np.int16))])

    train_set = datasets.JHUBrainDataset(train_dir.glob('*.pkl'), transforms=train_composed)
    val_set = datasets.JHUBrainInferDataset(val_dir.glob('*.pkl'), transforms=val_composed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    if half_precision:
        scaler = torch.cuda.amp.GradScaler()
    criterions = [nn.MSELoss(), losses.Grad3d(penalty='l2')]
    best_dsc = 0
    writer = SummaryWriter(log_dir=logs_dir)
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        avg_meter = utils.AverageMeter()
        model.train()
        for train_batch_idx, data in enumerate(train_loader):
            # if train_batch_idx > 2:
            #     break
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.to(device) for t in data]
            x = data[0]
            y = data[1]
            x_in = torch.cat((x,y), dim=1)
            with torch.cuda.amp.autocast():
                output = model(x_in)
                mse_loss = criterions[0](output[0], y)
                grad3d_loss = criterions[1](output[1], y)
                loss = mse_loss * weights[0] + grad3d_loss * weights[1]
                avg_meter.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            if half_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            writer.add_scalar('MSE/train_x_to_y', mse_loss.item(), train_writer_step)
            writer.add_scalar('Grad3d/train_x_to_y', grad3d_loss.item(), train_writer_step)
            writer.add_scalar('Loss/train_x_to_y', loss.item(), train_writer_step)

            del x_in
            del output
            # flip fixed and moving images
            loss = 0
            x_in = torch.cat((y, x), dim=1)
            with torch.cuda.amp.autocast():
                output = model(x_in)
                mse_loss = criterions[0](output[0], x)
                grad3d_loss = criterions[1](output[1], x)
                loss = mse_loss * weights[0] + grad3d_loss * weights[1]
                avg_meter.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            if half_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            writer.add_scalar('MSE/train_y_to_x', mse_loss.item(), train_writer_step)
            writer.add_scalar('Grad3d/train_y_to_x', grad3d_loss.item(), train_writer_step)
            writer.add_scalar('Loss/train_y_to_x', loss.item(), train_writer_step)
            train_writer_step += 1

            print(f"Batch {train_batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}, MSE: {mse_loss.item()/2:.6f}, Reg: {grad3d_loss.item()/2:.6f}")
        writer.add_scalar('Loss/train', avg_meter.avg, epoch)


        print('Epoch {} loss {:.4f}'.format(epoch, avg_meter.avg))
        '''
        Validation
        '''
        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            for val_batch_idx, data in enumerate(val_loader):
                # if val_batch_idx > 2:
                #     break
                model.eval()
                data = [t.to(device) for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                x_in = torch.cat((x, y), dim=1)
                grid_img = mk_grid_img(8, device, 1, config.img_size)
                with torch.cuda.amp.autocast():
                    output = model(x_in)
                    mse_loss = criterions[0](output[0], y)
                    grad3d_loss = criterions[1](output[1], y)
                    loss = mse_loss * weights[0] + grad3d_loss * weights[1]

                    def_out = reg_model([x_seg.to(device).float(), output[1].to(device)])
                    def_grid = reg_model_bilin([grid_img.to(device).float(), output[1].to(device)])
                    dsc = utils.dice_val(def_out.long(), y_seg.long(), 46)
                writer.add_scalar('MSE/val', mse_loss.item(), val_writer_step)
                writer.add_scalar('Grad3d/val', grad3d_loss.item(), val_writer_step)
                writer.add_scalar('Loss/val', loss.item(), val_writer_step)
                val_writer_step += 1
                writer.add_scalar('DSC/val', dsc.item(), epoch)
                eval_dsc.update(dsc.item(), x.size(0))
                print(f"Batch {val_batch_idx}/{len(val_loader)} Loss: {loss.item():.4f}, MSE: {mse_loss.item()/2:.6f}, Reg: {grad3d_loss.item()/2:.6f}, DSC: {dsc.item():.4f}")


        if eval_dsc.avg > best_dsc:
            print(f"New best DSC: {eval_dsc.avg=} > {best_dsc=}\nSaving model...")
            best_dsc = max(eval_dsc.avg, best_dsc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_dsc': best_dsc,
                'optimizer': optimizer.state_dict(),
            }, save_dir=experiment_dir, filename='best.pth.tar'.format(eval_dsc.avg))
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
        avg_meter.reset()
    writer.close()

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

def mk_grid_img(grid_step, device, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).to(device)
    return grid_img

def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, os.path.join(save_dir, filename))

if __name__ == '__main__':
    # GPU configuration
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print('Number of GPUs: ' + str(n_gpus))
        for GPU_idx in range(n_gpus):
            GPU_name = torch.cuda.get_device_name(GPU_idx)
            print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    running_on_dgx = os.path.isdir('/raid/')
    main(running_on_dgx)