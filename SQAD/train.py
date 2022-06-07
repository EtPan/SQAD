import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse

from utility import *
from hsi_setup import Engine, train_options, make_dataset

sigmas = [30, 50, 70]

def train_options(parser):
    def _parse_str_args(args):
        str_args = args.split(',')
        parsed_args = []
        for str_arg in str_args:
            arg = int(str_arg)
            if arg >= 0:
                parsed_args.append(arg)
        return parsed_args    
    parser.add_argument('--type', '-t', type=str, default='gauss', choices=['gauss', 'complex'])
    parser.add_argument('--prefix', '-p', type=str, default='denoise', help='prefix')
    parser.add_argument('--arch', '-a', metavar='ARCH', required=True, choices=model_names, help='model architecture: ' +' | '.join(model_names))
    parser.add_argument('--batchSize', '-b', type=int, default=16, help='training batch size. default=16')         
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate. default=1e-3.')
    parser.add_argument('--wd', type=float, default=0, help='weight decay. default=0')
    parser.add_argument('--loss', type=str, default='l2', help='which loss to choose.', choices=['l1', 'l2'])
    parser.add_argument('--init', type=str, default='kn', help='which init scheme to choose.', choices=['kn', 'ku', 'xn', 'xu', 'edsr'])
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda?')
    parser.add_argument('--no-log', action='store_true', help='disable logger?')
    parser.add_argument('--seed', type=int, default=2018, help='random seed to use. default=2018')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--no-ropt', '-nro', action='store_true', help='not resume optimizer')          
    parser.add_argument('--chop', action='store_true', help='forward chop')                                      
    parser.add_argument('--resumePath', '-rp', type=str, default=None, help='checkpoint to use.')
    parser.add_argument('--dataroot', '-d', type=str, default='./data/ICVL64_31.db', help='data root')
    parser.add_argument('--clip', type=float, default=1e6)
    parser.add_argument('--gpu-ids', type=str, default='2', help='gpu ids')
    opt = parser.parse_args()
    opt.gpu_ids = _parse_str_args(opt.gpu_ids)
    return opt


def make_train_dataset(opt, train_transform, target_transform, common_transform, batch_size=None, repeat=1):
    dataset = LMDBDataset(opt.dataroot, repeat=repeat)
    dataset = TransformDataset(dataset, common_transform)
    train_dataset = ImageTransformDataset(dataset, train_transform, target_transform)
    train_loader = DataLoader(train_dataset,batch_size=batch_size or opt.batchSize, shuffle=True,
                              num_workers=8, pin_memory=not opt.no_cuda, worker_init_fn=worker_init_fn)
    return train_loader

def make_valid_dataset(basefolder,mat_names):
    mat_datasets = [MatDataFromFolder(os.path.join(basefolder, name), size=5) for name in mat_names]
    mat_transform = Compose([LoadMatHSI(input_key='input', gt_key='gt',transform=lambda x:x[:, ...][None]),])                    
    mat_datasets = [TransformDataset(mat_dataset, mat_transform)for mat_dataset in mat_datasets]
    mat_loaders = [DataLoader(mat_dataset,batch_size=1, shuffle=False,num_workers=1, pin_memory=opt.no_cuda) for mat_dataset in mat_datasets] 
    return mat_loaders

def train_gauss(engine, base_lr, train_gauss_loaders_1, train_gauss_loaders_2, valid_loaders, valid_names):
    adjust_learning_rate(engine.optimizer, base_lr)
    while engine.epoch < 50:
        np.random.seed() 

        if engine.epoch == 30:
            adjust_learning_rate(engine.optimizer, base_lr*0.1)
        
        if engine.epoch == 40:
            adjust_learning_rate(engine.optimizer, base_lr*0.01)
        
        if engine.epoch <= 30:
            engine.train(train_gauss_loaders_1)
            engine.validate(valid_loaders[1], valid_names[1])
        else:
            engine.train(train_gauss_loaders_2)
            engine.validate(valid_loaders[0], valid_names[0])
            engine.validate(valid_loaders[1], valid_names[1])
        
        print('Latest Result Saving...')
        model_latest_path = os.path.join(engine.basedir, engine.prefix, 'model_latest.pth')
        engine.save_checkpoint(model_out_path=model_latest_path)

        display_learning_rate(engine.optimizer)
        if engine.epoch % epoch_per_save == 0:
            engine.save_checkpoint()

def train_complex(engine, base_lr, train_complex_loaders, valid_loaders, valid_names):
    adjust_learning_rate(engine.optimizer, base_lr)
    while engine.epoch < 100:
        np.random.seed()

        if engine.epoch == 70:
            adjust_learning_rate(engine.optimizer, base_lr*0.1)
        
        if engine.epoch == 80:
            adjust_learning_rate(engine.optimizer, base_lr*0.01)
        
        engine.train(train_complex_loaders)

        engine.validate(valid_loaders[2], valid_names[2])
        engine.validate(valid_loaders[3], valid_names[3])

        display_learning_rate(engine.optimizer)
        print('Latest Result Saving...')
        model_latest_path = os.path.join(engine.basedir, engine.prefix, 'model_latest.pth')
        engine.save_checkpoint(
            model_out_path=model_latest_path)

        display_learning_rate(engine.optimizer)
        if engine.epoch % epoch_per_save == 0:
            engine.save_checkpoint()

if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(description='Hyperspectral Image Denoising')
    opt = train_options(parser)
    print(opt)

    """Setup Engine"""
    engine = Engine(opt)

    """Dataset Setting"""
    HSI2Tensor = partial(HSI2Tensor)

    add_noniid_noise = Compose([AddNoiseNoniid(sigmas),SequentialSelect(transforms=[lambda x: x,
                                AddNoiseImpulse(),
                                AddNoiseStripe(),
                                AddNoiseDeadline()])])   
    
    common_transform_1 = lambda x: x
    common_transform_2 = Compose([partial(rand_crop, cropx=32, cropy=32),])

    target_transform = HSI2Tensor()
    
    train_transform_1 = Compose([AddNoise(50),HSI2Tensor()])
    train_transform_2 = Compose([AddNoiseBlind([30, 50, 70]),HSI2Tensor()])
    train_transform_3 = Compose([add_noniid_noise,HSI2Tensor()])

    train_gauss_loaders_1 = make_train_dataset(opt, train_transform_1, target_transform, common_transform_1, 8)
    train_gauss_loaders_2 = make_train_dataset(opt, train_transform_2, target_transform, common_transform_2, 32)
    train_complex_loaders = make_train_dataset(opt, train_transform_3, target_transform, common_transform_2, 32)


    basefolder = './data/'
    valid_names = ['icvl_512_30', 'icvl_512_50','icvl_512_noniid', 'icvl_512_mixture']
    
    valid_loaders = make_valid_dataset(basefolder,valid_names)        

    base_lr = opt.lr
    epoch_per_save = 10
    
    if opt.type == 'gauss':
        train_gauss(engine, base_lr, train_gauss_loaders_1, train_gauss_loaders_2, valid_loaders, valid_names,epoch_per_save)
    else: 
        train_complex(engine, base_lr, train_complex_loaders, valid_loaders, valid_names,epoch_per_save)
    
    
    
    
    
    