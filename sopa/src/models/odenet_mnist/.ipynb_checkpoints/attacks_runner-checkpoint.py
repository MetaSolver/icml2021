import foolbox as fb
import eagerpy as ep

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets


import numpy as np
import glob
import pandas as pd
from collections import defaultdict

from torchdiffeq._impl.rk_common import _ButcherTableau

import os
import sys
sys.path.append('/workspace/home/jgusak/neural-ode-sopa')

from sopa.src.solvers.rk_parametric_old import rk_param_tableau, odeint_plus
import dataloaders

from MegaAdversarial.src.utils.runner import  test, fix_seeds
from MegaAdversarial.src.attacks import (
    Clean,
    FGSM,
    PGD,
    LabelSmoothing,
    AdversarialVertex,
    AdversarialVertexExtra,
    FeatureScatter,
    NCEScatter,
    NCEScatterWithBuffer,
    MetaAttack
)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='')
parser.add_argument('--models_root', type=str, default='')
parser.add_argument('--save_path', type=str, default='')
parser.add_argument('--key_path_word', type=str, default='')

parser.add_argument('--min_eps', type=float, default=0.)
parser.add_argument('--max_eps', type=float, default=0.3)
parser.add_argument('--num_eps', type=int, default=20)
parser.add_argument('--epsilons', type=lambda s: [float(eps) for eps in s.split(',')], default=None)

parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])

args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

def run_set_of_attacks(epsilons, attack_modes, loaders, models_root, save_path = None, device = 'cuda', key_path_word = 'u2'):
    fix_seeds()

    df = pd.DataFrame()
    
    if key_path_word == 'u2':
        meta_model_path = "{}/*/*/*.pth".format(models_root)
    elif key_path_word == 'euler':
        meta_model_path = "{}/*/*.pth".format(models_root)
    
    i = 0
    for state_dict_path in glob.glob(meta_model_path, recursive = True):
        if not  (key_path_word in state_dict_path):
            continue
            
        print(state_dict_path)
        model, model_args = load_model(state_dict_path)
        model.eval()
        model.cuda()

        robust_accuracies = run_attack(model, epsilons, attack_modes, loaders, device)   
        robust_accuracies = {k : np.array(v) for k,v in robust_accuracies.items()}

        data = [list(dict(model_args._get_kwargs()).values()) +\
                list(robust_accuracies.values()) +\
                [epsilons]
               ]
        columns = list(dict(model_args._get_kwargs()).keys()) +\
                  list(robust_accuracies.keys()) +\
                  ['epsilons']
        
        df_tmp = pd.DataFrame(data = data, columns = columns) 
        df = df.append(df_tmp)
    
        if save_path is not None:
            df.to_csv(save_path, index = False)
            
        i += 1
        print('{} models have been processed'.format(i))
#         if i >=2:
#             break


def run_attack(model, epsilons, attack_modes, loaders, device='cuda'):
    robust_accuracies = defaultdict(list)
    
    for mode in attack_modes:
        for epsilon in epsilons:
#             CONFIG_PGD_TEST = {"eps": epsilon, "lr": 2.0 / 255 * 10, "n_iter": 20}
            CONFIG_PGD_TEST = {"eps": epsilon, "lr": 1, "n_iter": 5}
            CONFIG_FGSM_TEST = {"eps": epsilon}

            if mode == "clean":
                test_attack = Clean(model)
            elif mode == "fgsm":
                test_attack = FGSM(model, **CONFIG_FGSM_TEST)

            elif mode == "at":
                test_attack = PGD(model, **CONFIG_PGD_TEST)

            elif mode == "at_ls":
                test_attack = PGD(model, **CONFIG_PGD_TEST) # wrong func, fix this

            elif mode == "av":
                test_attack = PGD(model, **CONFIG_PGD_TEST) # wrong func, fix this

            elif mode == "fs":
                test_attack = PGD(model, **CONFIG_PGD_TEST) # wrong func, fix this

            print("Attack {}".format(mode))    
            test_metrics = test(loaders["val"], model, test_attack, device, show_progress=True)
            test_log = f"Test: | " + " | ".join(
                map(lambda x: f"{x[0]}: {x[1]:.6f}", test_metrics.items())
            )
            print(test_log)
            
            robust_accuracies['accuracy_{}'.format(mode)].append(test_metrics['accuracy_adv'])
      
    return robust_accuracies


def load_model(path):
    
    (_, state_dict), (_, model_args) = torch.load(path, map_location = 'cpu').items()
    print('\n', model_args)
    
    # Define a model given model_args
    is_odenet = model_args.network == 'odenet'

    if model_args.downsampling_method == 'conv':
        downsampling_layers = [
            nn.Conv2d(1, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        ]
    elif model_args.downsampling_method == 'res':
        downsampling_layers = [
            nn.Conv2d(1, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ]
        

    device = 'cpu'
    feature_layers = [ODEBlock(ODEfunc(64), model_args, device)] if is_odenet else [ResBlock(64, 64) for _ in range(6)]
    fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]

    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)
    
    # Initialize the model given a state_dict
    model.load_state_dict(state_dict) 
    
    return model, model_args


# Model utils
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc, model_args, device = 'cpu'):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        
        # make trainable parameters as attributes of ODE block,
        # recompute tableau at each forward step
        self.step_size = model_args.step_size

        self.method = model_args.method
        self.fix_param = None
        self.parameterization = None
        self.u0, self.v0 = None, None
        self.u_, self.v_ = None, None
        self.u, self.v = None, None
        
        
        self.eps = torch.finfo(torch.float32).eps
        self.device = device

        
        if self.method in ['rk4_param', 'rk3_param']:
            self.fix_param = model_args.fix_param
            self.parameterization = model_args.parameterization
            self.u0 = model_args.u0

            if self.fix_param:
                self.u = torch.tensor(self.u0)

                if self.parameterization == 'uv':
                    self.v0 = model_args.v0
                    self.v = torch.tensor(self.v0)

            else:
                # an important issue about putting leaf variables to the device https://discuss.pytorch.org/t/tensor-to-device-changes-is-leaf-causing-cant-optimize-a-non-leaf-tensor/37659
                self.u_ = nn.Parameter(torch.tensor(self.u0)).to(self.device)
                self.u = torch.clamp(self.u_, self.eps, 1. - self.eps).detach().requires_grad_(True)
                
                if self.parameterization == 'uv':
                    self.v0 = model_args.v0
                    self.v_ = nn.Parameter(torch.tensor(self.v0)).to(self.device)
                    self.v = torch.clamp(self.v_, self.eps, 1. - self.eps).detach().requires_grad_(True)

#             logger.info('Init | u {} | v {}'.format(self.u.data, (self.v if self.v is None else self.v.data)))

            self.alpha, self.beta, self.c_sol = rk_param_tableau(self.u, self.v, device = self.device,
                                                                 parameterization=self.parameterization,
                                                                 method = self.method)
            self.tableau = _ButcherTableau(alpha = self.alpha,
                                           beta = self.beta,
                                           c_sol = self.c_sol,
                                           c_error = torch.zeros((len(self.c_sol),), device = self.device))
    
    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        
        if self.method in ['rk4_param', 'rk3_param']:
            out = odeint_plus(self.odefunc, x, self.integration_time,
                              method=self.method, options = {'tableau':self.tableau, 'step_size':self.step_size})
        else:
            out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol,
                         method=self.method, options = {'step_size':self.step_size})
                
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


if __name__=="__main__":
    loaders = dataloaders.get_loader(batch_size=256,
                                     data_name='mnist',
                                     data_root=args.data_root,
                                     num_workers = 4,
                                     train=False, val=True)
    if args.epsilons is not None:
        epsilons = args.epsilons
    else:
        epsilons = np.linspace(args.min_eps, args.max_eps, num=args.num_eps) 
        
    run_set_of_attacks(epsilons=epsilons,
                   attack_modes = ["fgsm", "at", "at_ls", "av", "fs"][:1],
                   loaders = loaders,
                   models_root = args.models_root,
                   save_path = args.save_path,
                   key_path_word = args.key_path_word)


# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python3 odenet_plus_mnist_valyas_attacks.py --data_root "/workspace/raid/data/datasets" --models_root "/workspace/raid/data/jgusak/neural-ode-sopa/odenet_mnist/" --save_path './rk4_param_robust_accuracies_part4.csv' --min_eps 0.228 --max_eps 0.3 --num_eps 20 --key_path_word 'u2'

# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python3 odenet_plus_mnist_valyas_attacks.py --data_root "/workspace/raid/data/datasets" --models_root "/workspace/raid/data/jgusak/neural-ode-sopa/odenet_mnist/" --save_path './euler_robust_accuracies_part8.csv' --min_eps 0.51 --max_eps 0.6 --num_eps 10 --key_path_word 'euler' && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python3 odenet_plus_mnist_valyas_attacks.py --data_root "/workspace/raid/data/datasets" --models_root "/workspace/raid/data/jgusak/neural-ode-sopa/odenet_mnist/" --save_path './euler_robust_accuracies_part5.csv' --min_eps 0.21 --max_eps 0.3 --num_eps 10 --key_path_word 'euler' && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 python3 odenet_plus_mnist_valyas_attacks.py --data_root "/workspace/raid/data/datasets" --models_root "/workspace/raid/data/jgusak/neural-ode-sopa/odenet_mnist/" --save_path './euler_robust_accuracies_part2.csv' --min_eps 0.056 --max_eps 0.1 --num_eps 10 --key_path_word 'euler'


# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 python3 odenet_plus_mnist_valyas_attacks.py --data_root "/workspace/raid/data/datasets" --models_root "/workspace/raid/data/jgusak/neural-ode-sopa/odenet_mnist/" --save_path './euler_robust_accuracies_part7.csv' --min_eps 0.41 --max_eps 0.5 --num_eps 10 --key_path_word 'euler' && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 python3 odenet_plus_mnist_valyas_attacks.py --data_root "/workspace/raid/data/datasets" --models_root "/workspace/raid/data/jgusak/neural-ode-sopa/odenet_mnist/" --save_path './euler_robust_accuracies_part4.csv' --min_eps 0.156 --max_eps 0.2 --num_eps 10 --key_path_word 'euler' && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 python3 odenet_plus_mnist_valyas_attacks.py --data_root "/workspace/raid/data/datasets" --models_root "/workspace/raid/data/jgusak/neural-ode-sopa/odenet_mnist/" --save_path './euler_robust_accuracies_part1.csv' --min_eps 0. --max_eps 0.05 --num_eps 10 --key_path_word 'euler'

# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python3 odenet_plus_mnist_valyas_attacks.py --data_root "/workspace/raid/data/datasets" --models_root "/workspace/raid/data/jgusak/neural-ode-sopa/odenet_mnist/" --save_path './euler_robust_accuracies_part6.csv' --min_eps 0.31 --max_eps 0.4 --num_eps 10 --key_path_word 'euler' && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python3 odenet_plus_mnist_valyas_attacks.py --data_root "/workspace/raid/data/datasets" --models_root "/workspace/raid/data/jgusak/neural-ode-sopa/odenet_mnist/" --save_path './euler_robust_accuracies_part3.csv' --min_eps 0.106 --max_eps 0.15 --num_eps 10 --key_path_word 'euler'


# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python3 odenet_plus_mnist_valyas_attacks.py --data_root "/workspace/raid/data/datasets" --models_root "/workspace/raid/data/jgusak/neural-ode-sopa/odenet_mnist/" --save_path '/workspace/home/jgusak/neural-ode-sopa/experiments/odenet_mnist/results/robust_accuracies_fgsm/rk4_param_robust_accuracies_part1.csv' --epsilons 0.15,0.3,0.5  --key_path_word 'u2'
