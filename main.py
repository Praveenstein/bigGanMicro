import os
import pandas as pd
import random
from collections import OrderedDict
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torchvision.utils import make_grid
import warnings
warnings.filterwarnings("ignore")
#from torchsummaryX import summary


class ClassConditionalBN(nn.Module):
    def __init__(self, input_size, output_size, eps=1e-4, momentum=0.1):
        super(ClassConditionalBN, self).__init__()
        self.output_size, self.input_size = output_size, input_size
        # Prepare gain and bias layers
        self.gain = spectral_norm(nn.Linear(input_size, output_size, bias=False), eps=1e-4)
        self.bias = spectral_norm(nn.Linear(input_size, output_size, bias=False), eps=1e-4)
        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum

        self.register_buffer('stored_mean', torch.zeros(output_size))
        self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x, y):
        # Calculate class-conditional gains and biases
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                           self.training, 0.1, self.eps)
        return out * gain + bias

    def extra_repr(self):
        s = 'out: {output_size}, in: {input_size},'
        return s.format(**self.__dict__)


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation=nn.ReLU(inplace=False)):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class GeneratorResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=None, embed_dim=128, dim_z=384):
        super(GeneratorResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = self.in_channels // 4

        self.conv1 = spectral_norm(nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=1, padding=0),
                                   eps=1e-4)
        self.conv2 = spectral_norm(nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
                                   eps=1e-4)
        self.conv3 = spectral_norm(nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
                                   eps=1e-4)
        self.conv4 = spectral_norm(nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=1, padding=0),
                                   eps=1e-4)

        self.bn1 = ClassConditionalBN((3 * embed_dim) + dim_z, self.in_channels)
        self.bn2 = ClassConditionalBN((3 * embed_dim) + dim_z, self.hidden_channels)
        self.bn3 = ClassConditionalBN((3 * embed_dim) + dim_z, self.hidden_channels)
        self.bn4 = ClassConditionalBN((3 * embed_dim) + dim_z, self.hidden_channels)

        self.activation = nn.ReLU(inplace=False)

        self.upsample = upsample

    def forward(self, x, y):
        # Project down to channel ratio
        h = self.conv1(self.activation(self.bn1(x, y)))
        # Apply next BN-ReLU
        h = self.activation(self.bn2(h, y))
        # Drop channels in x if necessary
        if self.in_channels != self.out_channels:
            x = x[:, :self.out_channels]
            # Upsample both h and x at this point
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        # 3x3 convs
        h = self.conv2(h)
        h = self.conv3(self.activation(self.bn3(h, y)))
        # Final 1x1 conv
        h = self.conv4(self.activation(self.bn4(h, y)))
        return h + x


class Generator(nn.Module):
    def __init__(self, G_ch=64, dim_z=384, bottom_width=4, img_channels=1,
                 init='N02', n_classes_temp=7, n_classes_time=8, n_classes_cool=4, embed_dim=128):
        super(Generator, self).__init__()
        self.ch = G_ch
        self.dim_z = dim_z
        self.bottom_width = bottom_width
        self.init = init
        self.img_channels = img_channels

        self.embed_temp = nn.Embedding(n_classes_temp, embed_dim)
        self.embed_time = nn.Embedding(n_classes_time, embed_dim)
        self.embed_cool = nn.Embedding(n_classes_cool, embed_dim)

        self.linear = spectral_norm(nn.Linear(dim_z + (3 * embed_dim), 16 * self.ch * (self.bottom_width ** 2)),
                                    eps=1e-4)

        self.blocks = nn.ModuleList([
            GeneratorResBlock(16 * self.ch, 16 * self.ch),
            GeneratorResBlock(16 * self.ch, 16 * self.ch, upsample=nn.Upsample(scale_factor=2)),
            GeneratorResBlock(16 * self.ch, 16 * self.ch),
            GeneratorResBlock(16 * self.ch, 8 * self.ch, upsample=nn.Upsample(scale_factor=2)),
            GeneratorResBlock(8 * self.ch, 8 * self.ch),
            GeneratorResBlock(8 * self.ch, 8 * self.ch, upsample=nn.Upsample(scale_factor=2)),
            GeneratorResBlock(8 * self.ch, 8 * self.ch),
            GeneratorResBlock(8 * self.ch, 4 * self.ch, upsample=nn.Upsample(scale_factor=2)),
            Self_Attn(4 * self.ch),
            GeneratorResBlock(4 * self.ch, 4 * self.ch),
            GeneratorResBlock(4 * self.ch, 2 * self.ch, upsample=nn.Upsample(scale_factor=2)),
            GeneratorResBlock(2 * self.ch, 2 * self.ch),
            GeneratorResBlock(2 * self.ch, self.ch, upsample=nn.Upsample(scale_factor=2))
        ])

        self.final_layer = nn.Sequential(
            nn.BatchNorm2d(self.ch),
            nn.ReLU(inplace=False),
            spectral_norm(nn.Conv2d(self.ch, self.img_channels, kernel_size=3, padding=1)),
            nn.Tanh()
        )

        self.init_weights()

    def init_weights(self):
        print(f"Weight initialization : {self.init}")
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    torch.nn.init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    torch.nn.init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    torch.nn.init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print("Param count for G's initialized parameters: %d Million" % (self.param_count / 1000000))

    def forward(self, z, y_temp, y_time, y_cool):
        y_temp = self.embed_temp(y_temp)
        y_time = self.embed_time(y_time)
        y_cool = self.embed_cool(y_cool)
        z = torch.cat([z, y_temp, y_time, y_cool], 1)
        # First linear layer
        h = self.linear(z)
        # Reshape
        h = h.view(h.size(0), -1, self.bottom_width, self.bottom_width)
        # Loop over blocks
        for i, block in enumerate(self.blocks):
            if i != 8:
                h = block(h, z)
            else:
                h = block(h)
        # Apply batchnorm-relu-conv-tanh at output
        h = self.final_layer(h)
        return h


class DiscriminatorResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, preactivation=True,
                 downsample=None, channel_ratio=4):
        super(DiscriminatorResBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels // channel_ratio
        self.preactivation = preactivation
        self.activation = nn.ReLU(inplace=False)
        self.downsample = downsample

        # Conv layers
        self.conv1 = spectral_norm(nn.Conv2d(self.in_channels, self.hidden_channels,
                                             kernel_size=1, padding=0), eps=1e-4)
        self.conv2 = spectral_norm(nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
                                   eps=1e-4)
        self.conv3 = spectral_norm(nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
                                   eps=1e-4)
        self.conv4 = spectral_norm(nn.Conv2d(self.hidden_channels, self.out_channels,
                                             kernel_size=1, padding=0), eps=1e-4)

        self.learnable_sc = True if (in_channels != out_channels) else False
        if self.learnable_sc:
            self.conv_sc = spectral_norm(nn.Conv2d(in_channels, out_channels - in_channels,
                                                   kernel_size=1, padding=0), eps=1e-4)

    def shortcut(self, x):
        if self.downsample:
            x = self.downsample(x)
        if self.learnable_sc:
            x = torch.cat([x, self.conv_sc(x)], 1)
        return x

    def forward(self, x):
        # 1x1 bottleneck conv
        h = self.conv1(F.relu(x))
        # 3x3 convs
        h = self.conv2(self.activation(h))
        h = self.conv3(self.activation(h))
        # relu before downsample
        h = self.activation(h)
        # downsample
        if self.downsample:
            h = self.downsample(h)
            # final 1x1 conv
        h = self.conv4(h)
        return h + self.shortcut(x)


class Discriminator(nn.Module):
    def __init__(self, D_ch=64, img_channels=1, init='N02', n_classes_temp=7, n_classes_time=8, n_classes_cool=4):
        super(Discriminator, self).__init__()
        self.ch = D_ch
        self.init = init
        self.img_channels = img_channels
        self.output_dim = n_classes_temp + n_classes_time + n_classes_cool + 2

        # Prepare model
        # Stem convolution
        self.input_conv = spectral_norm(nn.Conv2d(self.img_channels, self.ch, kernel_size=3, padding=1), eps=1e-4)

        self.blocks = nn.Sequential(
            DiscriminatorResBlock(self.ch, 2 * self.ch, downsample=nn.AvgPool2d(2)),
            DiscriminatorResBlock(2 * self.ch, 2 * self.ch),
            DiscriminatorResBlock(2 * self.ch, 4 * self.ch, downsample=nn.AvgPool2d(2)),
            DiscriminatorResBlock(4 * self.ch, 4 * self.ch),
            Self_Attn(4 * self.ch),
            DiscriminatorResBlock(4 * self.ch, 8 * self.ch, downsample=nn.AvgPool2d(2)),
            DiscriminatorResBlock(8 * self.ch, 8 * self.ch),
            DiscriminatorResBlock(8 * self.ch, 8 * self.ch, downsample=nn.AvgPool2d(2)),
            DiscriminatorResBlock(8 * self.ch, 8 * self.ch),
            DiscriminatorResBlock(8 * self.ch, 16 * self.ch, downsample=nn.AvgPool2d(2)),
            DiscriminatorResBlock(16 * self.ch, 16 * self.ch),
            DiscriminatorResBlock(16 * self.ch, 16 * self.ch, downsample=nn.AvgPool2d(2)),
            DiscriminatorResBlock(16 * self.ch, 16 * self.ch),
        )
        # Linear output layer. The output dimension is typically 1, but may be
        # larger if we're e.g. turning this into a VAE with an inference output
        self.linear = spectral_norm(nn.Linear(16 * self.ch, self.output_dim), eps=1e-4)

        self.init_weights()

    def init_weights(self):
        print(f"Weight initialization : {self.init}")
        self.param_count = 0
        for module in self.modules():
            if (isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
                if self.init == 'ortho':
                    torch.nn.init.orthogonal_(module.weight)
                elif self.init == 'N02':
                    torch.nn.init.normal_(module.weight, 0, 0.02)
                elif self.init in ['glorot', 'xavier']:
                    torch.nn.init.xavier_uniform_(module.weight)
                else:
                    print('Init style not recognized...')
                self.param_count += sum([p.data.nelement() for p in module.parameters()])
        print("Param count for D's initialized parameters: %d Million" % (self.param_count / 1000000))

    def forward(self, x):
        # Run input conv
        h = self.input_conv(x)
        # Blocks
        h = self.blocks(h)
        # Apply global sum pooling as in SN-GAN
        h = torch.sum(nn.ReLU(inplace=False)(h), [2, 3])
        # Get initial class-unconditional output
        out = self.linear(h)
        # Get projection of final featureset onto class vectors and add to evidence
        # out = out + torch.sum(self.embed_temp(y_temp) * h, 1, keepdim=True) + torch.sum(self.embed_time(y_time) * h, 1, keepdim=True) + torch.sum(self.embed_cool(y_cool) * h, 1, keepdim=True)
        return out


class DiffAugment:
    def __init__(self, policy='color,translation,cutout', channels_first=True):
        self.policy = policy
        print(f'Diff. Augment Policy : {policy}')
        self.channels_first = channels_first
        self.AUGMENT_FNS = {'color': [self.rand_brightness, self.rand_saturation, self.rand_contrast],
                            'translation': [self.rand_translation],
                            'cutout': [self.rand_cutout]}

    def __call__(self, x):
        if self.policy:
            if not self.channels_first:
                x = x.permute(0, 3, 1, 2)
            for p in self.policy.split(','):
                for f in self.AUGMENT_FNS[p]:
                    x = f(x)
            if not self.channels_first:
                x = x.permute(0, 2, 3, 1)
            x = x.contiguous()
        return x

    def rand_brightness(self, x):
        x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
        return x

    def rand_saturation(self, x):
        x_mean = x.mean(dim=1, keepdim=True)
        x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
        return x

    def rand_contrast(self, x):
        x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
        x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
        return x

    def rand_translation(self, x, ratio=0.125):
        shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
        translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(x.size(2), dtype=torch.long, device=x.device),
            torch.arange(x.size(3), dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
        grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
        x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
        x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
        return x

    def rand_cutout(self, x, ratio=0.5):
        cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
        offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
        offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
        grid_batch, grid_x, grid_y = torch.meshgrid(
            torch.arange(x.size(0), dtype=torch.long, device=x.device),
            torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
            torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
        )
        grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
        grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
        mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
        mask[grid_batch, grid_x, grid_y] = 0
        x = x * mask.unsqueeze(1)
        return x


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset for temp , time and cool
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        weights = torch.DoubleTensor(weights)
        self.weights = weights

    def _get_label(self, dataset, idx):
        return dataset[idx][4]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class MicrographDataset(Dataset):
    """
    A custom Dataset class for Micrograph data which returns the following
    # Micrograph image
    # Inputs : Anneal Temperature , Anneal Time and Type of cooling used
    ------------------------------------------------------------------------------------
    Attributes

    df : pandas.core.frame.DataFrame
        A Dataframe that contains the proper entries (i.e. dataframe corresponding to new_metadata.xlsx)
    root_dir : str
        The path of the folder where the images are located
    transform : torchvision.transforms.transforms.Compose
        The transforms that are to be applied to the loaded images
    """

    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.transform = transform
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        temp_dict = {970: 0, 800: 1, 900: 2, 1100: 3, 1000: 4, 700: 5, 750: 6}
        time_dict = {90: 0, 1440: 1, 180: 2, 5: 3, 480: 4, 5100: 5, 60: 6, 2880: 7}
        microconst_dict = {'spheroidite': 0, 'network': 1, 'spheroidite+widmanstatten': 2, 'pearlite+spheroidite': 3,
                           'pearlite': 4, 'pearlite+widmanstatten': 5}
        cooling_dict = {'Q': 0, 'FC': 1, 'AR': 2, '650-1H': 3}
        row = self.df.loc[idx]
        img_name = row['path']
        img_path = self.root_dir + '/' + 'Cropped' + img_name
        anneal_temp = temp_dict[row['anneal_temperature']]
        if row['anneal_time_unit'] == 'H':
            anneal_time = int(row['anneal_time']) * 60
        else:
            anneal_time = row['anneal_time']
        anneal_time = time_dict[anneal_time]
        cooling_type = cooling_dict[row['cool_method']]
        microconst = microconst_dict[row['primary_microconstituent']]
        img = Image.open(img_path)
        img = img.convert('L')
        if self.transform:
            img = self.transform(img)
        return img, anneal_temp, anneal_time, cooling_type, microconst


class MicrographBigGAN(pl.LightningModule):
    def __init__(self, root_dir, df_dir, batch_size, augment_bool=True, lr=0.0002,
                 n_classes_temp=7, n_classes_time=8, n_classes_cool=4):
        super().__init__()
        self.save_hyperparameters()
        self.root_dir = root_dir
        self.df_dir = df_dir
        self.generator = Generator(G_ch=64)
        self.discriminator = Discriminator(D_ch=64)
        self.diffaugment = DiffAugment()
        self.augment_bool = augment_bool
        self.batch_size = batch_size
        self.lr = lr
        self.n_classes_temp = n_classes_temp
        self.n_classes_time = n_classes_time
        self.n_classes_cool = n_classes_cool

    def forward(self, z, y_temp, y_time, y_cool):
        return self.generator(z, y_temp, y_time, y_cool)

    def multilabel_categorical_crossentropy(self, y_true, y_pred, margin=0., gamma=1.):
        """ y_true: positive=1, negative=0, ignore=-1
        """
        y_true = y_true.clamp(-1, 1)
        if len(y_pred.shape) > 2:
            y_true = y_true.view(y_true.shape[0], 1, 1, -1)
            _, _, h, w = y_pred.shape
            y_true = y_true.expand(-1, h, w, -1)
            y_pred = y_pred.permute(0, 2, 3, 1)

        y_pred = y_pred + margin
        y_pred = y_pred * gamma

        y_pred[y_true == 1] = -1 * y_pred[y_true == 1]
        y_pred[y_true == -1] = -1e12

        y_pred_neg = y_pred.clone()
        y_pred_neg[y_true == 1] = -1e12

        y_pred_pos = y_pred.clone()
        y_pred_pos[y_true == 0] = -1e12

        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        return neg_loss + pos_loss

    def Omni_Dloss(self, disc_real, disc_fake, y_temp_real, y_time_real, y_cool_real):
        b = y_temp_real.shape[0]
        y_temp = F.one_hot(y_temp_real, num_classes=self.n_classes_temp).to(self.device)
        y_time = F.one_hot(y_time_real, num_classes=self.n_classes_time).to(self.device)
        y_cool = F.one_hot(y_cool_real, num_classes=self.n_classes_cool).to(self.device)
        y_real = torch.cat([y_temp, y_time, y_cool, torch.tensor([1, 0]).repeat(b, 1).to(self.device)], 1).float()
        y_real.requires_grad = True
        y_fake = torch.cat([torch.zeros((b, self.n_classes_temp + self.n_classes_time + self.n_classes_cool)),
                            torch.tensor([0, 1]).repeat(b, 1)], 1).float().to(self.device)
        y_fake.requires_grad = True
        d_loss_real = self.multilabel_categorical_crossentropy(y_true=y_real, y_pred=disc_real)
        d_loss_fake = self.multilabel_categorical_crossentropy(y_true=y_fake, y_pred=disc_fake)
        d_loss = d_loss_real.mean() + d_loss_fake.mean()
        return d_loss

    def Omni_Gloss(self, disc_fake, y_temp_fake, y_time_fake, y_cool_fake):
        b = y_temp_fake.shape[0]
        y_temp = F.one_hot(y_temp_fake, num_classes=self.n_classes_temp).to(self.device)
        y_time = F.one_hot(y_time_fake, num_classes=self.n_classes_time).to(self.device)
        y_cool = F.one_hot(y_cool_fake, num_classes=self.n_classes_cool).to(self.device)
        y_fake_g = torch.cat([y_temp, y_time, y_cool, torch.tensor([1, 0]).repeat(b, 1).to(self.device)], 1).float()
        y_fake_g.requires_grad = True
        g_loss = self.multilabel_categorical_crossentropy(y_true=y_fake_g, y_pred=disc_fake)
        return g_loss.mean()

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, y_temp_real, y_time_real, y_cool_real, _ = batch
        z = torch.randn(real.shape[0], 384)
        z = z.type_as(real)
        y_temp_fake = torch.randint(self.n_classes_temp, (real.shape[0],)).to(self.device)
        y_time_fake = torch.randint(self.n_classes_time, (real.shape[0],)).to(self.device)
        y_cool_fake = torch.randint(self.n_classes_cool, (real.shape[0],)).to(self.device)

        if optimizer_idx == 0:
            fake = self(z, y_temp_fake, y_time_fake, y_cool_fake)

            if self.augment_bool:
                disc_real = self.discriminator(self.diffaugment(real))
                disc_fake = self.discriminator(self.diffaugment(fake))
            else:
                disc_real = self.discriminator(real)
                disc_fake = self.discriminator(fake)
            d_loss = self.Omni_Dloss(disc_real, disc_fake, y_temp_real, y_time_real, y_cool_real)
            # print(d_loss.requires_grad)
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        if optimizer_idx == 1:
            fake = self(z, y_temp_fake, y_time_fake, y_cool_fake)
            if self.augment_bool:
                disc_fake = self.discriminator(self.diffaugment(fake))
            else:
                disc_fake = self.discriminator(fake)
            g_loss = self.Omni_Gloss(disc_fake, y_temp_fake, y_time_fake, y_cool_fake)
            # print(g_loss.requires_grad)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        opt_g = optim.Adam(self.generator.parameters(), lr=self.lr, weight_decay=0.001)
        opt_d = optim.Adam(self.discriminator.parameters(), lr=self.lr, weight_decay=0.0005)
        return (
            {'optimizer': opt_d, 'frequency': 4},
            {'optimizer': opt_g, 'frequency': 1}
        )

    def train_dataloader(self):
        img_transforms = transforms.Compose([
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize([0.5 for _ in range(1)], [0.5 for _ in range(1)]),
        ])
        df = pd.read_excel(self.df_dir, engine='openpyxl')
        dataset = MicrographDataset(df, self.root_dir, transform=img_transforms)
        return DataLoader(dataset, sampler=ImbalancedDatasetSampler(dataset),
                          batch_size=self.batch_size, shuffle=False)



ROOT_DIR = '../input/highcarbon-micrographs/For Training/Cropped'
DF_DIR = '../input/highcarbon-micrographs/new_metadata.xlsx'

gan = MicrographBigGAN(ROOT_DIR, DF_DIR, batch_size=12)

trainer = pl.Trainer(max_epochs=600, gpus=1 if torch.cuda.is_available() else 0, accumulate_grad_batches=8)

trainer.fit(gan)

trainer.save_checkpoint("MicroGAN_checkpoint.ckpt")

torch.save(gan.generator.state_dict(), 'BigGAN-deep.pth')
