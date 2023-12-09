import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio
from matplotlib.pyplot import imread, imsave
from skimage.transform import resize
import time
import sys
from experiments.uniform_denoising.models import *
sys.path.append('../')

from admm_utils import *
from torch import optim
from models import *
import math
import glob
from experiments.uniform_denoising.models.BN_Net import *
from experiments.uniform_denoising.models.util import *
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def run(f_name, specific_result_dir, noise_sigma, num_iter, rho, sigma_0, L):

    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    img = imread(f_name)[:,:,:3]
    if img.dtype == 'uint8':
        img = img.astype('float32') / 255  # scale to [0, 1]
    elif img.dtype == 'float32':
        img = img.astype('float32')
    else:
        raise TypeError()
    
    img = np.clip(resize(img, (128, 128)), 0, 1)
    imsave(specific_result_dir + 'true.png', img)
    img = img.transpose((2, 0, 1))
    x_true = torch.from_numpy(img).unsqueeze(0).type(dtype)


    b = x_true
    b = b + noise_sigma * (2 * torch.rand(b.shape) - 1).type(dtype) # add uniform noise with mean 0
    b_clipped = torch.clamp(b, 0, 1)
    imsave(specific_result_dir + 'corrupted.png',
           b_clipped.reshape(1, 3, 128, 128)[0].permute((1, 2, 0)).cpu().numpy()) ## CHANGED

    def fn(x):
        return torch.norm(x - b) ** 2 / 2

    G = skip(3, 3,
             num_channels_down=[16, 32, 64, 128, 128],
             num_channels_up=[16, 32, 64, 128, 128],  # [16, 32, 64, 128, 128],
             num_channels_skip=[0, 0, 0, 0, 0],
             filter_size_up=3, filter_size_down=3, filter_skip_size=1,
             upsample_mode='nearest',  # downsample_mode='avg',
             need1x1_up=False,
             need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').type(dtype)
    #z = torch.zeros_like(x_true).type(dtype).normal_()
    CNN_weight_name = [
        '1.0.1.1.weight', '1.0.1.1.bias',
        '1.1.1.1.weight', '1.1.1.1.bias',
        '1.1.4.1.weight', '1.1.4.1.bias',
        '3.1.weight', '3.1.bias',
        '6.1.weight', '6.1.bias',
        '9.1.weight', '9.1.bias']
    G.to(device)
    # 这段代码的目的是冻结神经网络中某些层的权重参数，使其不参与梯度更新，然后计算并打印剩余可训练参数的数量。
    for name, param in G.named_parameters():
        ### here we do not want to fix the BN
        if name in CNN_weight_name:
            param.requires_grad = False
    net_total_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    print('After freezing, the left trainable param is {}'.format(net_total_params))
    # z = torch.zeros_like(x_true).type(dtype).normal_()

    ## BN net
    bnnet = BNNet(3)
    bnnet.to(device)

    # 创建一个与输入数据具有相同大小的张量，用于存储噪声。
    noise_like = torch.empty(1, 3, 128, 128).to(device)
    # 使用与输入数据相同大小的零张量，并使用高斯分布（正态分布）生成随机数，然后乘以 0.1 以控制噪声的强度。这将产生一个具有小幅度随机值的张量，表示高斯噪声。
    g_noise = torch.zeros_like(noise_like).normal_() * 1e-1
    # 设置生成的噪声张量为需要梯度计算，这意味着在网络的训练过程中，模型会根据输入和这个噪声张量的梯度进行调整。
    g_noise.requires_grad = True
    # 将生成的高斯噪声张量添加到参数列表 p_c 中，以便后续将其添加到模型中。
    # p_c = [g_noise]

    x = G(g_noise).clone().detach()
    scaled_lambda_ = torch.zeros_like(x, requires_grad=False).type(dtype)

    x.requires_grad, g_noise.requires_grad = False, True

    # since we use exact minimization over x, we don't need the grad of x
    #z.requires_grad = False
    opt_z = optim.Adam(G.parameters(), lr=L)

    sigma_0 = torch.tensor(sigma_0).type(dtype)
    Gz = G(g_noise)

    record = {"psnr_gt": [],
              "mse_gt": [],
              "total_loss": [],
              "prior_loss": [],
              "fidelity_loss": [],
              "cpu_time": [],
              }

    results = None
    for t in range(num_iter):
        G.train()
        bnnet.train()
        # for x
        with torch.no_grad():
            x = linf_proj(Gz.detach() - scaled_lambda_, b, noise_sigma)
            # x = Gz.detach() - scaled_lambda_

        # for z (GD)
        opt_z.zero_grad()
        optimizer_noise.zero_grad()
        optimizer_bn.zero_grad()

        g_noise_input = bnnet(g_noise)
        Gz = G(g_noise_input)
        loss_z = torch.norm(b- Gz) ** 2 / 2 + (rho / 2) * torch.norm(x - G(z) + scaled_lambda_) ** 2
        loss_z.backward()
        opt_z.step()
        optimizer_noise.step()
        optimizer_bn.step()
        # for dual var(lambda)
        with torch.no_grad():
            Gz = G(g_noise_input).detach()
            x_Gz = x - Gz
            scaled_lambda_.add_(sigma_0 * x_Gz)

        if results is None:
            results = Gz.detach()
        else:
            results = results * 0.99 + Gz.detach() * 0.01

        psnr_gt = peak_signal_noise_ratio(x_true.cpu().numpy(), results.cpu().numpy())
        mse_gt = np.mean((x_true.cpu().numpy() - results.cpu().numpy()) ** 2)
        fidelity_loss = fn(torch.tensor(results).cuda()).detach()
        
        #if (t + 1) % 1000 == 0:
            #imsave(specific_result_dir + 'iter%d_PSNR_%.2f.png' % (t, psnr_gt), results[0].cpu().numpy().transpose((1, 2, 0)))

        record["psnr_gt"].append(psnr_gt)
        record["mse_gt"].append(mse_gt)
        record["fidelity_loss"].append(fidelity_loss.item())
        record["cpu_time"].append(time.time())

        if (t + 1) % 10 == 0:
            print('Img %d Iteration %5d PSRN_gt: %.2f MSE_gt: %e' % (f_num, t + 1, psnr_gt, mse_gt))
    np.savez(specific_result_dir + 'record', **record)

# torch.manual_seed(500)
if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

dataset_dir = '../../data/'
results_dir = '../../data/results/DRP_DIP_linf_ball/'

os.makedirs(results_dir, exist_ok=True)
f_name_list = glob.glob('../../data/*016.jpg')

for f_num, f_name in enumerate(f_name_list):
    tp = '8'
    specific_result_dir = results_dir + tp + '/'
    os.makedirs(specific_result_dir, exist_ok=True)
    run(f_name=f_name,
        specific_result_dir=specific_result_dir,
        noise_sigma=25/255,
        num_iter=10000,
        rho=1,
        sigma_0=1,
        L=0.001)
