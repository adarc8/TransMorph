import glob
import os, losses, utils
from pathlib import Path

from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph

def main(running_on_dgx):
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    if running_on_dgx:
        database_dir = Path('/raid/data/users/adarc/registration/IXI/IXI_data')
    else:
        database_dir = Path(r'D:\Datasets\IXI\ixi_from_transmorph\IXI_data')  #
    test_dir = database_dir / 'Test'
    model_idx = -1
    weights = [1, 0.02]
    model_folder = 'TransMorph_mse_{}_diffusion_{}/'.format(weights[0], weights[1])
    model_dir = 'experiments/' + model_folder
    dict = utils.process_label()
    if os.path.exists('experiments/'+model_folder[:-1]+'.csv'):
        os.remove('experiments/'+model_folder[:-1]+'.csv')
    # csv_writter(model_folder[:-1], 'experiments/' + model_folder[:-1])
    line = ''
    # for i in range(46):
    #     line = line + ',' + dict[i]
    # csv_writter(line, 'experiments/' + model_folder[:-1])

    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)
    best_model = torch.load('/raid/data/users/adarc/registration/downloads/IXI_checkpoint_from_git_TransMorph_Validation_dsc0.744.pth.tar')['state_dict']
    model.load_state_dict(best_model)
    model.to(device)
    reg_model = utils.register_model(device, (160, 192, 224), 'nearest')
    reg_model.to(device)
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.JHUBrainInferDataset(glob.glob((test_dir / '*.pkl').as_posix()), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)
    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        for test_batch_idx, data in enumerate(test_loader):
            model.eval()
            data = [t.to(device) for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            if test_batch_idx > 2:
                break

            x_in = torch.cat((x,y),dim=1)
            plt.imshow(y[0,0,:,80,:].detach().cpu().numpy())
            plt.title('fixed')
            plt.show()
            # ploting flow rgb
            flowc = copy.deepcopy(flow.cpu().detach().numpy())[0].transpose(1,2,3,0)
            #normalize
            flowc[:,:,:,0] = (flowc[:,:,:,0] - np.min(flowc[:,:,:,0]))/(np.max(flowc[:,:,:,0]) - np.min(flowc[:,:,:,0]))
            flowc[:,:,:,1] = (flowc[:,:,:,1] - np.min(flowc[:,:,:,1]))/(np.max(flowc[:,:,:,1]) - np.min(flowc[:,:,:,1]))
            flowc[:,:,:,2] = (flowc[:,:,:,2] - np.min(flowc[:,:,:,2]))/(np.max(flowc[:,:,:,2]) - np.min(flowc[:,:,:,2]))

            plt.imshow(flowc[:, 80])
            plt.title(r'$\phi$ - registration field')
            plt.show()

            x_def, flow = model(x_in)
            #
            def_out = reg_model([x_seg.to(device).float(), flow.to(device)])
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            line = utils.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            line = line #+','+str(np.sum(jac_det <= 0)/np.prod(tar.shape))
            csv_writter(line, 'experiments/' + model_folder[:-1])
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(tar.shape)))
            dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), 46)
            dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 46)
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            stdy_idx += 1

            # flip moving and fixed images
            y_in = torch.cat((y, x), dim=1)
            y_def, flow = model(y_in)
            def_out = reg_model([y_seg.cuda().float(), flow.cuda()])
            tar = x.detach().cpu().numpy()[0, 0, :, :, :]

            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            line = utils.dice_val_substruct(def_out.long(), x_seg.long(), stdy_idx)
            line = line #+ ',' + str(np.sum(jac_det < 0) / np.prod(tar.shape))
            out = def_out.detach().cpu().numpy()[0, 0, :, :, :]
            print('det < 0: {}'.format(np.sum(jac_det <= 0)/np.prod(tar.shape)))
            csv_writter(line, 'experiments/' + model_folder[:-1])
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            dsc_trans = utils.dice_val(def_out.long(), x_seg.long(), 46)
            dsc_raw = utils.dice_val(y_seg.long(), x_seg.long(), 46)
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(), dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            stdy_idx += 1

        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

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