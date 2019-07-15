# Unsupervised person generation with semantic parsing transformation (CVPR 2019)

# Demo model for DeepFashion (256x256)
# The model consists of a generator, a general discriminator and a face discriminator
# Loss function: adversarial loss (general + face) + pose consistency loss + content loss + semantic aware loss
# The model is trained in a progressive manner using fadein strategy

import torch
import sys
import itertools
from .base_model import BaseModel
from . import networks
from .VGG_multi import VGG16, vgg_model
from .Normalization import Normalization, normalization_model
from .GramMatrix_bodypart import patch_gram_matrix
from models import GlobalLocalPRDemo_models as GlobalLocal_models
from util.pose_transform import AffineTransformLayer

class PoseGuidedGenerationDemoModel(BaseModel):
    def name(self):
        return 'PoseGuidedGenerationDemoModel'

    
    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
        else:
            
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_pose_det(self, load_path):

        net = self.netpose_det

        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print(('loading the pose det model from %s' % load_path))

        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        del state_dict._metadata

        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        net.load_state_dict(state_dict)

    def load_netD(self, which_net, load_path):

        if which_net == 'netD':
            net = self.netD
        elif which_net ==  'netD_face':
            net = self.netD_face

        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        
        try:
            net.load_state_dict(torch.load(load_path))
        except:               
            
            model_dict = net.state_dict()         

            print(('loading the Discriminator from %s' % load_path))
            # if you are using PyTorch newer than 0.4 (e.g., built from
            # GitHub source), you can remove str() on self.device
            state_dict = torch.load(load_path, map_location=str(self.device))
            del state_dict._metadata

            # patch InstanceNorm checkpoints prior to 0.4
            for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))

            for k, v in list(state_dict.items()):                
                if k in model_dict:
                    print(('loading...%s'%k))
                    model_dict[k] = v

            net.load_state_dict(model_dict)


    def load_pretrain_global(self,save_path):
        network_label = 'Global_Generator'
        net = self.netG
        if isinstance(net, torch.nn.DataParallel):
            network = net.module
        try:
            network.load_state_dict(torch.load(save_path))
        except:   
            pretrained_dict = torch.load(save_path)                
            model_dict = network.state_dict()         

            for k, v in list(pretrained_dict.items()):
                kk = 'model_global.' + k                    
                if kk in model_dict:
                    print(('loading...%s'%kk))
                    model_dict[kk] = v

            if sys.version_info >= (3,0):
                not_initialized = set()
            else:
                from sets import Set
                not_initialized = Set()                    

            for k, v in list(model_dict.items()):
                kk = '.'.join(k.split('model_global.')[1:])
                if kk not in pretrained_dict or v.size() != pretrained_dict[kk].size():
                    not_initialized.add(k)
            print(('Pretrained network %s has fewer layers; The following are not initialized:' % network_label))
            print((sorted(not_initialized)))
            network.load_state_dict(model_dict)        

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specifify the training losses you want to print out
        self.loss_names = ['pose_det_A', 'pose_det_B', 'G_A', 'D_A', 'G_B', 'D_B', 'pose_det', 'content_loss', 'patch_style', 'D_A_face', 'D_B_face']

        # specify the models you want to save to the dist.
        if self.isTrain:
            self.model_names = ['G', 'D', 'D_face']
            self.fade_in_alpha = 0
        else:
            self.model_names = ['G']
            self.fade_in_alpha = 1

        # load/define networks

        if self.isTrain:

            # Code for pose detector
            self.netpose_det = networks.define_G(opt.input_nc, opt.pose_cnt,
                                              opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
            
            # Load the well trained pose detector
            load_pose_path = '/export/home/ssj/pytorch-CycleGAN-and-pix2pix/checkpoints/pose_detector/latest_net_pose_det.pth'
            self.load_pose_det(load_pose_path)

            # Network for perceptual loss
            self.vgg = vgg_model(self.gpu_ids)

            cnn_normalization_mean = [0.485, 0.456, 0.406]
            cnn_normalization_std = [0.229, 0.224, 0.225]
            self.normalization = normalization_model(cnn_normalization_mean, cnn_normalization_std, self.gpu_ids)

            # Extract face region
            self.face_layer = AffineTransformLayer(1, (256,256), None)
            self.face_layer = torch.nn.DataParallel(self.face_layer, self.gpu_ids)

       
        # Use the structure of Global and Local Generator 
        # input channel: img_channel(3) + parsing(9) + pose (18) + parsing (9) + pose (18)
        self.netG = GlobalLocal_models.define_G(3+9+18+9+18, 18, (256,256),  (64, 128, 256, 512, 512, 512),(512, 512, 512, 256, 128, 3),  'mask', opt.init_type, True, self.gpu_ids)

        # Downsample the input pose map
        self.downsample = torch.nn.AvgPool2d(2, stride=2).to(self.gpu_ids[0])
        self.downsample = torch.nn.DataParallel(self.downsample, self.gpu_ids)
            
        if self.isTrain:
            pretrain_path = opt.global_pretrain_path
            self.load_pretrain_global(pretrain_path)

            use_sigmoid = opt.no_lsgan

            # here we use the D without fadein 256 x 256, but the weights are loaded from the D trained on 128 x 128
            self.netD = networks.define_D(opt.output_nc + 9, opt.ndf, opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            
            self.netD_face = networks.define_D_fadein(opt.output_nc, opt.ndf, opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

            load_path_netD = '/export/home/ssj/pytorch-CycleGAN-and-pix2pix/checkpoints/pose_generation_gram_def_global128_face/latest_net_D.pth'
            load_path_netD_face = '/export/home/ssj/pytorch-CycleGAN-and-pix2pix/checkpoints/pose_generation_gram_def_global128_face/latest_net_D_face.pth'            
            self.load_netD('netD', load_path_netD)
            self.load_netD('netD_face', load_path_netD_face)

            # define loss functions

            # pose consistency loss
            self.criterionPose = torch.nn.MSELoss()

            # adversarial loss
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)

            # Use l2 as content loss
            self.criterionIdt = torch.nn.MSELoss()

            # Use l2 as semantic-aware loss
            self.criterionSty = torch.nn.MSELoss()

            # initialize optimizers
            self.optimizer_pose_det = torch.optim.Adam(self.netpose_det.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            # fix Global and finetune Local
            if opt.niter_fix_global > 0:
                params = []
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()
                params_dict = dict(self.netG.named_parameters())
                for key, value in list(params_dict.items()):
                    if key.find('model_global') == -1:
                        params += [value]
                        finetune_list.add(key)
                print(('-----only training the local generator network for %d epochs-----'%opt.niter_fix_global))
                print(('The layers that are finetuned are ', sorted(finetune_list)))

            else:
                params = list(self.netG.parameters())

            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1,0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters(), self.netD_face.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.optimizers.append(self.optimizer_pose_det)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def set_input(self, input):

        # inputs of 256 x 256
        self.real_A = input['A'].to(self.device)

        self.pose_A = input['A_pose'].to(self.device)
        self.pose_B = input['B_pose'].to(self.device)
        
        self.AtoB_warps = input['AtoB_warps'].to(self.device)
        self.AtoB_masks = input['AtoB_masks'].to(self.device) #B_masks

        self.BtoA_warps = input['BtoA_warps'].to(self.device)
        self.BtoA_masks = input['BtoA_masks'].to(self.device) #A_masks

        self.A_parsing = input['A_parsing'].to(self.device)
        self.B_parsing = input['B_parsing'].to(self.device)

        #inputs of 128 x 128 
        self.real_A_128 = input['A_128'].to(self.device)

        self.pose_A_128 = input['A_pose_128'].to(self.device)
        self.pose_B_128 = input['B_pose_128'].to(self.device)


        self.AtoB_warps_128 = input['AtoB_warps_128'].to(self.device)
        self.AtoB_masks_128 = input['AtoB_masks_128'].to(self.device)

        self.BtoA_warps_128 = input['BtoA_warps_128'].to(self.device)
        self.BtoA_masks_128 = input['BtoA_masks_128'].to(self.device)

        self.A_parsing_128 = input['A_parsing_128'].to(self.device)
        self.B_parsing_128 = input['B_parsing_128'].to(self.device)

        self.face_A_warp = input['face_A_warp'].to(self.device)
        self.face_B_warp = input['face_B_warp'].to(self.device)

        self.image_paths = input['A_paths']
        self.image_A_paths = input['A_paths']
        self.image_B_paths = input['B_paths']

    def set_fade_in_alpha(self, fade_in_alpha):
        self.fade_in_alpha = fade_in_alpha

    def forward(self):
        # Generate the image with A and B_pose
        self.input_G_A = torch.cat((self.real_A, self.A_parsing, self.pose_A, self.B_parsing, self.pose_B),1)
        self.down_input_G_A = torch.cat((self.real_A_128, self.A_parsing_128, self.pose_A_128, self.B_parsing_128, self.pose_B_128),1)
        self.fake_B = self.netG(self.input_G_A, self.down_input_G_A, self.AtoB_warps, self.AtoB_masks, self.AtoB_warps_128, self.AtoB_masks_128, self.fade_in_alpha)
        

        # Generate the image with B and A_pose
        self.input_G_B = torch.cat((self.fake_B, self.B_parsing, self.pose_B, self.A_parsing, self.pose_A),1)


        self.fake_B_128 = self.downsample(self.fake_B)
        self.down_input_G_B = torch.cat((self.fake_B_128, self.B_parsing_128, self.pose_B_128, self.A_parsing_128, self.pose_A_128),1)
       
        self.fake_A = self.netG(self.input_G_B, self.down_input_G_B, self.BtoA_warps, self.BtoA_masks, self.BtoA_warps_128, self.BtoA_masks_128, self.fade_in_alpha)

        if self.isTrain:
            # Train the pose detector

            # calculate the fake A pose
            self.fake_A_pose = self.netpose_det(self.fake_A)

            # calculate the fake B pose
            self.fake_B_pose = self.netpose_det(self.fake_B)

            # calculate the VGG feature for content loss and style loss
            self.normalization_fake_A = self.normalization(self.fake_A)
            self.normalization_real_A = self.normalization(self.real_A)
            self.normalization_fake_B = self.normalization(self.fake_B)

            self.fake_A_feat = self.vgg(self.normalization_fake_A) # batchsize x 128 x 128 x 128
            self.real_A_feat = self.vgg(self.normalization_real_A) # batchsize x 128 x 128 x 128
            self.fake_B_feat = self.vgg(self.normalization_fake_B) # batchsize x 128 x 128 x 128
            

            # Downsample the masks (face excluded)
            self.downsample_AtoB_masks = self.downsample(self.B_parsing[:,1:])
            self.downsample_BtoA_masks = self.downsample(self.A_parsing[:,1:])

            valid_AtoB_masks = torch.max(self.downsample_AtoB_masks, 2, True)[0]
            valid_AtoB_masks = torch.max(valid_AtoB_masks, 3, True)[0]

            valid_BtoA_masks = torch.max(self.downsample_BtoA_masks, 2, True)[0]
            valid_BtoA_masks = torch.max(valid_BtoA_masks, 3, True)[0]

            self.visibility = valid_BtoA_masks * valid_AtoB_masks
            
            # Extract the face from the whole image
            self.real_A_face = self.face_layer(self.real_A, self.face_A_warp, None)
            self.fake_A_face = self.face_layer(self.fake_A, self.face_A_warp, None)
            self.fake_B_face = self.face_layer(self.fake_B, self.face_B_warp, None)




    def backward_pose_det(self):
          
        self.loss_pose_det_A = self.criterionPose(self.fake_A_pose, self.pose_A) * 700 
        self.loss_pose_det_B = self.criterionPose(self.fake_B_pose, self.pose_B) * 700
        loss_pose_det = self.loss_pose_det_B + self.loss_pose_det_A
        return loss_pose_det
        

    def backward_D_basic(self, netD, real_img, fake_img, real_parsing, fake_parsing):
        # Real
        real = torch.cat((real_img, real_parsing), 1)
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        
        # Fake
        fake = torch.cat((fake_img, fake_parsing), 1)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_face(self, netD, real_img, fake_img):
        # Real
        pred_real = netD(real_img)
        loss_D_real = self.criterionGAN(pred_real, True)
        
        # Fake
        pred_fake = netD(fake_img.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):

        # Train the general discriminator
        self.loss_D_A = self.backward_D_basic(self.netD, self.real_A, self.fake_A, self.A_parsing, self.A_parsing)
        self.loss_D_B = self.backward_D_basic(self.netD, self.real_A, self.fake_B, self.A_parsing, self.B_parsing)

         # Train the face discriminator
        self.loss_D_A_face = self.backward_D_face(self.netD_face, self.real_A_face, self.fake_A_face)
        self.loss_D_B_face = self.backward_D_face(self.netD_face, self.real_A_face, self.fake_B_face)

  
    def backward_G(self):

        self.D_fake_B = torch.cat((self.fake_B, self.B_parsing), 1)
        self.D_fake_A = torch.cat((self.fake_A, self.A_parsing), 1)

        # Train the general discriminator, as well as the face discriminator
        self.loss_G_A = self.criterionGAN(self.netD(self.D_fake_B), True) + self.criterionGAN(self.netD_face(self.fake_B_face), True)
        self.loss_G_B = self.criterionGAN(self.netD(self.D_fake_A), True) + self.criterionGAN(self.netD_face(self.fake_A_face), True)


        # pose consistency loss
        self.loss_pose_det = self.backward_pose_det()

        # Using content loss (L2)
        self.loss_content_loss = 0.03 * self.criterionIdt(self.fake_A_feat, self.real_A_feat)

        # semantic-aware loss
        self.loss_patch_style_real_A_fake_B = self.criterionSty(self.visibility * patch_gram_matrix(self.fake_B_feat, self.downsample_AtoB_masks), 
                                                                self.visibility * patch_gram_matrix(self.real_A_feat, self.downsample_BtoA_masks))
        self.loss_patch_style_fake_A_fake_B = (self.visibility * patch_gram_matrix(self.fake_A_feat, self.downsample_BtoA_masks) - 
                                               self.visibility * patch_gram_matrix(self.fake_B_feat, self.downsample_AtoB_masks)) ** 2
        self.loss_patch_style_fake_A_fake_B = self.loss_patch_style_fake_A_fake_B.mean()

        
        self.loss_patch_style = self.loss_patch_style_fake_A_fake_B + self.loss_patch_style_real_A_fake_B 

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_pose_det +  self.loss_content_loss + self.loss_patch_style

        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()

        # freeze the pose detector
        self.set_requires_grad([self.netpose_det], False)
        self.set_requires_grad([self.downsample], False)

        # G
        self.set_requires_grad([self.netD, self.netD_face], False)
        self.optimizer_G.zero_grad()
        torch.nn.utils.clip_grad_norm(self.netG.parameters(), 100)
        self.backward_G()
        self.optimizer_G.step()

        # D
        self.set_requires_grad([self.netD, self.netD_face], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def update_fixed_params(self,opt):
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1,0.999))
        print('-----------Now also finetuning the global generator---------')


