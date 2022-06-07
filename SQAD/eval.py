import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import argparse

from utility import *
from hsi_setup import Engine, train_options
import models


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


prefix = 'test'

def test_syn(engine,basefolder,resfolder,names):
    
    for i in range(len(names)):
          datadir = os.path.join(basefolder, names[i])
          if resfolder==None:
              resdir = None
          else:     
              resdir = os.path.join(resfolder, names[i])
              if not os.path.exists(resdir):
                  os.mkdir(resdir)
          mat_dataset = MatDataFromFolder(datadir, size=None)
          #mat_dataset.filenames = [
          #         os.path.join(datadir, 'Lehavim_0910-1626.mat') 
                   #os.path.join(datadir, 'Lehavim_0910-1627.mat') 
          #     ]    
          mat_transform = Compose([
              LoadMatHSI(input_key='input', gt_key='gt',transform=lambda x:x[:,:,:][None]), # for validation
          ])
          mat_dataset = TransformDataset(mat_dataset, mat_transform)
          mat_loader = DataLoader(mat_dataset,batch_size=1, shuffle=False,
                    num_workers=1, pin_memory=cuda)   
          
          res_arr, input_arr = engine.test_develop(mat_loader, savedir=resdir, verbose=True)
          print(res_arr.mean(axis=0))


if __name__ == '__main__':
    """Training settings"""
    parser = argparse.ArgumentParser(
        description='Hyperspectral Image Denoising')
    opt = train_options(parser)
    print(opt)

    cuda = not opt.no_cuda
    opt.no_log = True

    """Setup Engine"""
    engine = Engine(opt)

    basefolder = '../mine/data/' #your input data dir
    resfolder = None #'./result/'#your result dir
    names_gauss = ['icvl_512_blind']#['icvl_512_30','icvl_512_50','icvl_512_70','icvl_512_blind']
    names_complex = ['icvl_512_noniid','icvl_512_stripe','icvl_512_deadline','icvl_512_impulse','icvl_512_mixture']

    #-------------test syn gauss
    test_syn(engine,basefolder,resfolder,names_gauss)
    #-------------test syn complex
    #test_syn(engine,basefolder,resfolder,names_complex)
    
    
