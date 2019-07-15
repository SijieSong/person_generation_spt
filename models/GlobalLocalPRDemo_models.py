# The encoder of local enhancer is from 32 -> 64
from util import pose_utils
from util.pose_transform import AffineTransformLayer

import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.modules.module import _addindent
import numpy as np


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print(('initialize network with %s' % init_type))
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net

def define_G(input_nc, pose_dim, image_size, nfilters_enc, nfilters_dec, warp_skip, init_type, use_input_pose=True, gpu_ids=[]):
    netG = None

    assert image_size == (256, 256)
    assert nfilters_enc == (64, 128, 256, 512, 512, 512)
    assert nfilters_dec == (512, 512, 512, 256, 128, 3)
    netG = Local_Generator(input_nc, pose_dim, image_size, nfilters_enc, nfilters_dec, warp_skip, use_input_pose)

    return init_net(netG, init_type, gpu_ids)




class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Cropping2D(nn.Module):
    def __init__(self, crop_size):
        super(Cropping2D, self).__init__()
        self.crop_size = crop_size
    def forward(self, input):
        return input[:,:,self.crop_size:-self.crop_size,self.crop_size:-self.crop_size]

class Block(nn.Module):
    def __init__(self,  input_nc, output_nc, down=True, bn=True, dropout=False, leaky=True):
        super(Block, self).__init__()
        self.net = self.build_net( input_nc, output_nc, down, bn, dropout, leaky)

    def build_net(self, input_nc, output_nc, down=True, bn=True, dropout=False, leaky=True):
        model = []
        if leaky:
            model.append(nn.LeakyReLU(0.2))
        else:
            model.append(nn.ReLU())
        if down:
            model.append(nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=False))
        else:
            model.append(nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, bias=False))
            model.append(Cropping2D(1))
        if bn:
            model.append(nn.InstanceNorm3d(1, eps=1e-3, affine=True, track_running_stats=False))
        if dropout:
            model.append(nn.Dropout2d())
        return nn.ModuleList(model)

    def forward(self, input):
        for module in self.net:
            if("Instance" in module.__class__.__name__):
                input = input.unsqueeze(1)
                
                input = module(input)
                
                input = input.squeeze(1)
               
            else:
                input = module(input)
        return input


class encoder(nn.Module):
    def __init__(self, input_nc, nfilters_enc):
        super(encoder, self).__init__()
        self.input_nc = input_nc
        self.nfilters_enc = nfilters_enc
        self.net = self.build_net(input_nc, nfilters_enc)

    def build_net(self, input_nc, nfilters_enc):
        model = []
        for i, nf in enumerate(nfilters_enc):
            if i == 0:
                model.append(nn.Conv2d(input_nc, nf, kernel_size=3, padding=1, bias=True))
            elif i == len(nfilters_enc) - 1:
                model.append(Block(nfilters_enc[i - 1], nf, bn=False))
            else:
                model.append(Block(nfilters_enc[i - 1], nf))
        return nn.ModuleList(model)

    def forward(self, input, fade_in_feat=0, fade_in_alpha=0):
        outputs = []
        for i,module in enumerate(self.net):
            if(i==0):
                # fade in the feat from high-resolution image, fade_in_alpha grows from 0 to 1
                out =  (1-fade_in_alpha) * module(input) + fade_in_alpha * fade_in_feat
                outputs.append(out)
            else:
                out = module(out)
                outputs.append(out)
        return outputs

class local_encoder(nn.Module):
    def __init__(self, input_nc, nfilters_enc):
        super(local_encoder, self).__init__()
        self.input_nc = input_nc
        self.nfilters_enc = nfilters_enc
        self.net = self.build_net(input_nc, nfilters_enc)

    def build_net(self, input_nc, nfilters_enc):
        model = []
        for i, nf in enumerate(nfilters_enc):
            if i == 0:
                # input_nc -> 64/2 = 32
                model.append(nn.Conv2d(input_nc, nf//2, kernel_size=3, padding=1, bias=True))
            # different from the global encoder, the last layer before the global encoder is after BN
            else:
                # 64/2 -> 128/2 => 32 -> 64 
                model.append(Block(nfilters_enc[i - 1]//2, nf//2))
        return nn.ModuleList(model)

    def forward(self, input):
        outputs = []
        for i,module in enumerate(self.net):
            if(i==0):
                out = module(input)
                outputs.append(out)
            else:
                out = module(out)
                outputs.append(out)
        return outputs


class decoder(nn.Module):
    def __init__(self, nfilters_dec, nfilters_enc, num_skips = 1):
        super(decoder, self).__init__()
        # number of skip connections
        self.num_skips = num_skips
        self.nfilters_dec = nfilters_dec
        self.nfilters_enc = nfilters_enc
        self.net = self.build_net(nfilters_dec)
        self.upsampling = nn.Upsample(scale_factor=2, mode='nearest')



    def build_net(self, nfilters_dec):
        model_dec = []
        for i, nf in enumerate(nfilters_dec):
            if i==0:
                model_dec.append(Block((self.num_skips)*self.nfilters_enc[-1], nf, down=False, leaky=False, dropout=True))
            elif 0 < i < 3:
                # due to skip connections
                model_dec.append(Block((self.num_skips)*self.nfilters_enc[-(i+1)] + nfilters_dec[i - 1], nf, down=False, leaky=False, dropout=True))
            elif i==len(nfilters_dec)-1:
                model_dec.append(nn.ReLU())
                model_dec.append(nn.Conv2d((self.num_skips)*self.nfilters_enc[-(i+1)] + nfilters_dec[i - 1], nf, kernel_size=3, padding=1, bias=True))
            else:
                # due to skip connections
                model_dec.append(Block((self.num_skips)*self.nfilters_enc[-(i+1)] + nfilters_dec[i - 1], nf, down=False, leaky=False))
            
        model_dec.append(nn.Tanh())

        return nn.ModuleList(model_dec)

    def forward(self, skips):
        for i in range(len(self.nfilters_dec)):
            if (i == 0):
                out = self.net[0](skips[-(i+1)])
            elif i<len(self.nfilters_dec)-1:
                out = torch.cat([out, skips[-(i+1)]], 1)
                out = self.net[i](out)
            else:
                #return the feat map of 128 x 128 x 128
                feat = out.clone()
                # final processing, upsampling
                out = torch.cat([out, skips[-(i+1)]], 1)
                # (128 + 64 + 64)  x 128 x 128 -> (128 + 64 + 64) x 256 x 256
                out = self.upsampling(out)       
                out = self.net[i](out)
                # 3 x 256 x 256
                out = self.net[i+1](out)
        # applying non linearity
        out = self.net[-1](out)
        #out = self.upsampling(out)
        return out, feat

class local_decoder(nn.Module):
    def __init__(self, nfilters_dec, nfilters_enc, num_skips = 1):
        super(local_decoder, self).__init__()
        # number of skip connections
        self.num_skips = num_skips
        self.nfilters_dec = nfilters_dec
        self.nfilters_enc = nfilters_enc
        self.net = self.build_net(nfilters_dec)


    # nfilter_dec:(128,3)
    # nfilter_enc: (64,128)
    def build_net(self, nfilters_dec):
        model_dec = []
        for i, nf in enumerate(nfilters_dec):
            if i==0:
                # Input: (64 + 64 + 128) x 128 x 128 -> Output: 128/2 x 256 x 256
                model_dec.append(Block((self.num_skips)*self.nfilters_enc[1]//2 + nfilters_dec[i], nf//2, down=False, leaky=False))          
            elif i==len(nfilters_dec)-1:
                # Input: (32 + 32 + 64) x 256 x 256 -> Ouptut: 3 x 256 x 256
                model_dec.append(nn.ReLU())
                model_dec.append(nn.Conv2d((self.num_skips)*self.nfilters_enc[0]//2 + nfilters_dec[i - 1]//2, nf, kernel_size=3, padding=1, bias=True))
        model_dec.append(nn.Tanh())

        return nn.ModuleList(model_dec)

    def forward(self, skips):
        #skips: [(32 + 32) x 256 x 256, (128 + 64 + 64) x 128 x 128]
        for i in range(len(self.nfilters_dec)):
            if (i == 0):
                out = self.net[0](skips[-(i+1)])
            else:
                # final processing
                out = torch.cat([out, skips[-(i+1)]], 1)
                out = self.net[i](out)
                out = self.net[i+1](out)
        # applying non linearity
        out = self.net[-1](out)
        return out


class Global_Generator(nn.Module):
    def __init__(self, input_nc, pose_dim, image_size, nfilters_enc, nfilters_dec, warp_skip, use_input_pose=True):
        super(Global_Generator, self).__init__()
        self.input_nc = input_nc
        # number of skip connections
        self.num_skips = 1 if warp_skip=='None' else 2
        self.warp_skip = warp_skip
        self.pose_dim = pose_dim
        self.nfilters_dec = nfilters_dec
        self.nfilters_enc = nfilters_enc
        self.image_size = image_size
        self.use_input_pose = use_input_pose
        # input parsing result to encoder_pose
        self.encoder_app = encoder(input_nc-self.pose_dim - 9, nfilters_enc)
        self.encoder_pose = encoder(self.pose_dim + 9, nfilters_enc)

        self.decoder = decoder(nfilters_dec, nfilters_enc, self.num_skips)
        self.pose_dim = 18


    def get_imgpose(self, input, use_input_pose, pose_dim):
        inp_img = input[:, :12] # include pose and parsing
        inp_pose = input[:, 12:12 + pose_dim] if use_input_pose else None

        tg_parsing = input[:, 12+pose_dim: 21+pose_dim] # target parsing
        tg_pose_index = 21 + pose_dim if use_input_pose else 6
        tg_pose = input[:, tg_pose_index:]
        
        return inp_img, inp_pose, tg_parsing, tg_pose

    def forward(self, input, warps, masks, fade_in_app, fade_in_pose, fade_in_alpha):
        
        inp_app, inp_pose, tg_parsing, tg_pose = self.get_imgpose(input, self.use_input_pose, self.pose_dim)
        inp_app = torch.cat([inp_app, inp_pose], dim=1)

        #fade in the feat from high resolution image
        skips_app = self.encoder_app(inp_app, fade_in_app, fade_in_alpha)
        #len(enc_filter), enc_c, h, w

        inp_pose = torch.cat([tg_pose, tg_parsing], dim=1)
        
        #fade in the feat from high resolution image        
        skips_pose = self.encoder_pose(inp_pose, fade_in_pose, fade_in_alpha)
        #len(enc_filter), enc_c, h, w

        # define concatenate func
        skips = self.concatenate_skips(skips_app, skips_pose, warps, masks)
        out, feat = self.decoder(skips)

        # return out and skips for local generator
        return out, feat, skips

    def concatenate_skips(self, skips_app, skips_pose, warps, masks):
        skips = []
        for i, (sk_app, sk_pose) in enumerate(zip(skips_app, skips_pose)):
            if i < 4:
                out = AffineTransformLayer(10 if self.warp_skip == 'mask' else 1, self.image_size, self.warp_skip)(sk_app, warps, masks)
                out = torch.cat([out, sk_pose], dim=1)
            else:
                out = torch.cat([sk_app, sk_pose], dim=1)
            skips.append(out)
        return skips


class Local_Generator(nn.Module):
    def __init__(self, input_nc, pose_dim, image_size, nfilters_enc, nfilters_dec, warp_skip, use_input_pose=True):
        super(Local_Generator, self).__init__()
        self.input_nc = input_nc
        # number of skip connections
        self.num_skips = 1 if warp_skip=='None' else 2
        self.warp_skip = warp_skip
        self.pose_dim = pose_dim
        self.nfilters_dec = nfilters_dec
        self.nfilters_enc = nfilters_enc
        self.image_size = image_size
        self.use_input_pose = use_input_pose
        self.pose_dim = 18

        # build global_generator

        ###### global generator model #####    
        self.model_global= Global_Generator(self.input_nc, self.pose_dim, (128,128), self.nfilters_enc, self.nfilters_dec, self.warp_skip, self.use_input_pose)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)


        # local enhance layers
        self.model_local_encoder_app = local_encoder(input_nc-self.pose_dim - 9, nfilters_enc[:2])
        self.model_local_encoder_pose = local_encoder(self.pose_dim + 9, nfilters_enc[:2])
        self.model_local_decoder = local_decoder(nfilters_dec[-2:], nfilters_enc, self.num_skips)
        

    def get_imgpose(self, input, use_input_pose, pose_dim):
        inp_img = input[:, :12] # include pose and parsing
        inp_pose = input[:, 12:12 + pose_dim] if use_input_pose else None

        tg_parsing = input[:, 12+pose_dim: 21+pose_dim] # target parsing
        tg_pose_index = 21 + pose_dim if use_input_pose else 6
        tg_pose = input[:, tg_pose_index:]
        
        return inp_img, inp_pose, tg_parsing, tg_pose

    def forward(self, input, down_input, warps, masks, warps_128, masks_128,fade_in_alpha):
        
        inp_app, inp_pose, tg_parsing, tg_pose = self.get_imgpose(input, self.use_input_pose, self.pose_dim)
  
        inp_app = torch.cat([inp_app, inp_pose], dim=1)
        local_skips_app = self.model_local_encoder_app(inp_app)
        #skips_app:[32 x 256 x 256, 64 x 128 x 128]

 
        inp_pose = torch.cat([tg_pose, tg_parsing], dim=1)
        local_skips_pose = self.model_local_encoder_pose(inp_pose)
        #skips_pose: [32 x 256 x 256, 64 x 128 x 128]

        # define concatenate func
        local_skips = self.concatenate_skips(local_skips_app, local_skips_pose, warps, masks)
        # local_skips: [(32 + 32) x 256 x 256, (64 + 64) x 128 x 128]

        # downsample input to feed global_generator
        global_output, global_feat, global_skips = self.model_global(down_input, warps_128, masks_128, local_skips_app[1], local_skips_pose[1], fade_in_alpha) 
        # 3 x 256 x 256, 128 x 128 x 128, [(64 + 64) x 128 x 128, ...]

        # Concate the output of global skips and global output
        local_skips[1] = torch.cat([global_feat,global_skips[0]], dim=1)
        #local_skips: [(32 + 32) x 256 x 256, (128 + 64 + 64) x 128 x 128]

        out = self.model_local_decoder(local_skips)

        out = fade_in_alpha * out + (1-fade_in_alpha) * global_output
        return out

    def concatenate_skips(self, skips_app, skips_pose, warps, masks):
        skips = []
        for i, (sk_app, sk_pose) in enumerate(zip(skips_app, skips_pose)):
            out = AffineTransformLayer(10 if self.warp_skip == 'mask' else 1, self.image_size, self.warp_skip)(sk_app, warps, masks)
            out = torch.cat([out, sk_pose], dim=1)
           
            skips.append(out)
        return skips

       




