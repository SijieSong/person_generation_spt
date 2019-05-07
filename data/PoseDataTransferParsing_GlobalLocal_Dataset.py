# Calculate the pose maps online from annotation files
# downsample images, including pose map, parsing map and raw images.
# downsample everything
import os.path
from data.base_dataset import BaseDataset
from data.base_dataset import get_transform_my as get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torchvision.transforms as transforms
import torch
import numpy
import pandas as pd
from util import pose_utils
from util import pose_transform


class PoseDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.dir_A = os.path.join(opt.dataroot)
        self.dir_B = os.path.join(opt.dataroot)

        self.annotation_file = pd.read_csv(opt.annotation_file, sep=':')
        self.annotation_file = self.annotation_file.set_index('name')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        if self.opt.isTrain:
            f_A = open('./data/train_A_parsing.txt','r')
            f_B = open('./data/train_B_parsing.txt','r')
            self.A_paths = [_.split('\r\n')[0] for _ in f_A.readlines()]
            self.B_paths = [_.split('\r\n')[0] for _ in f_B.readlines()]
            f_A.close()
            f_B.close()

        else:
            f_A = open('./data/test_A_example.txt', 'r')
            f_B = open('./data/test_B_example.txt', 'r')
            self.A_paths = [_.split('\r\n')[0] for _ in f_A.readlines()]
            self.B_paths = [_.split('\r\n')[0] for _ in f_B.readlines()]
            f_A.close()
            f_B.close()

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.transform = get_transform(opt)
        self.down_transform = get_transform(opt)
        self._warp_skip = 'mask'
        self.pose_dim = 18
        self._image_size = (256,256)

    def compute_cord_warp(self, kp_array1, kp_array2, img_size=(256,256)):
        if self._warp_skip == 'full':
            warp = [numpy.empty([1, 8]), 1]
        else:
            warp = [numpy.empty([10, 8]),
                    numpy.empty([10] + list(img_size))]


        if self._warp_skip == 'mask':
            warp[0] = pose_transform.affine_transforms(kp_array1, kp_array2, self.pose_dim)
            warp[1] = pose_transform.pose_masks(kp_array2, img_size, self.pose_dim)
        else:
            warp[0] = pose_transform.estimate_uniform_transform(kp_array1, kp_array2, self.pose_dim)

        return warp

    def compute_cord_face_warp(self, kp_array1):
        face_warp = pose_transform.estimate_face_transform(kp_array1, self.pose_dim, self._image_size)
        return face_warp

    def downsample_pose_array(self, input_array):
        kp_array = input_array.copy()
        for idx in xrange(18):
            if kp_array[idx,0] != -1 and kp_array[idx,1] != -1 :
                kp_array[idx,0] = int(kp_array[idx,0] * 0.5)
                kp_array[idx,1] = int(kp_array[idx,1] * 0.5)

        return kp_array

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        
        # if self.opt.serial_batches:
        if True:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        print('(A, B) = (%d, %d)' % (index, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        downsample_A_img = A_img.resize((128,128),Image.BICUBIC)
        downsample_B_img = B_img.resize((128,128),Image.BICUBIC)

        A_img_name = A_path.split('/')[-1]

        A_row = self.annotation_file.loc[A_img_name]
        A_kp_array = pose_utils.load_pose_cords_from_strings(A_row['keypoints_y'], A_row['keypoints_x'])
	  
       
        #downsample pose location
        downsample_A_kp_array = self.downsample_pose_array(A_kp_array)


        A_pose = pose_utils.cords_to_map(A_kp_array, self._image_size)
        A_pose = numpy.transpose(A_pose,(2, 0, 1)) # A_pose: 256 x 256 x 18 => 18 x 256 x 256
        A_pose = torch.Tensor(A_pose)
       
        downsample_A_pose = pose_utils.cords_to_map(downsample_A_kp_array, (128,128))
        downsample_A_pose = numpy.transpose(downsample_A_pose,(2, 0, 1)) 
        downsample_A_pose = torch.Tensor(downsample_A_pose)


        B_img_name = B_path.split('/')[-1]

        B_row = self.annotation_file.loc[B_img_name]
        B_kp_array = pose_utils.load_pose_cords_from_strings(B_row['keypoints_y'], B_row['keypoints_x'])
        
        #downsample pose location
        downsample_B_kp_array = self.downsample_pose_array(B_kp_array)
        
        B_pose = pose_utils.cords_to_map(B_kp_array, (self._image_size))
        B_pose = numpy.transpose(B_pose,(2, 0, 1)) # B_pose: 256 x 256 x 18 => 18 x 256 x 256
        B_pose = torch.Tensor(B_pose)

        downsample_B_pose = pose_utils.cords_to_map(downsample_B_kp_array, (128,128))
        downsample_B_pose = numpy.transpose(downsample_B_pose,(2, 0, 1)) # B_pose: 256 x 256 x 18 => 18 x 256 x 256
        downsample_B_pose = torch.Tensor(downsample_B_pose)


        A = self.transform(A_img)
        downsample_A = self.transform(downsample_A_img)
        downsample_B = self.transform(downsample_B_img)

        AtoB_warps, AtoB_masks = self.compute_cord_warp(A_kp_array, B_kp_array)
        BtoA_warps, BtoA_masks = self.compute_cord_warp(B_kp_array, A_kp_array)

        downsample_AtoB_warps, downsample_AtoB_masks = self.compute_cord_warp(downsample_A_kp_array, downsample_B_kp_array, img_size=(128,128))
        downsample_BtoA_warps, downsample_BtoA_masks = self.compute_cord_warp(downsample_B_kp_array, downsample_A_kp_array, img_size=(128,128))

        # Compute the face warp
        face_A_warp = self.compute_cord_face_warp(A_kp_array)
        face_B_warp = self.compute_cord_face_warp(B_kp_array)


        A_parsing_file = './parsing/'+ A_img_name.split('.jpg')[0] + '.npy'
	# A_parsing_file = '/data/songsijie/deepfashion_parsing/merge_parsing/' + A_img_name.split('.jpg')[0] + '.npy'
        A_parsing_data = numpy.load(A_parsing_file)
        A_parsing = numpy.zeros((9,256,256),dtype='int8')
        for id in xrange(9):
            A_parsing[id] = (A_parsing_data == id+1).astype('int8')


        B_parsing_file = './parsing/'+ B_img_name.split('.jpg')[0] + '.npy'
        # B_parsing_file = '/data/songsijie/deepfashion_parsing/merge_parsing/' + B_img_name.split('.jpg')[0] + '.npy'
        B_parsing_data = numpy.load(B_parsing_file)
        B_parsing = numpy.zeros((9,256,256),dtype='int8')
        for id in xrange(9):
            B_parsing[id] = (B_parsing_data == id+1).astype('int8')


        ####Downsample A_parsing & B_parsing
        # the parsing size became 9 x 128 x 128 
        [X, Y] = numpy.meshgrid(range(0,256,2),range(0,256,2))
        downsample_A_parsing = A_parsing[:,Y,X]
        downsample_B_parsing = B_parsing[:,Y,X]


        input_nc = self.opt.input_nc
        output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        #A: Image A
        #B_pose: B_pose
        #A_pose: A_pose

        return {'A': A, 
                'A_pose': A_pose,  'B_pose': B_pose, 
                'AtoB_warps': AtoB_warps.astype('float32'), 'BtoA_warps': BtoA_warps.astype('float32'),
                'AtoB_masks': AtoB_masks.astype('float32'), 'BtoA_masks': BtoA_masks.astype('float32'),
                'A_parsing': A_parsing.astype('float32'), 'B_parsing': B_parsing.astype('float32'),

                'A_128': downsample_A, 'B_128': downsample_B, 
                'A_pose_128': downsample_A_pose, 'B_pose_128': downsample_B_pose,          
                'AtoB_warps_128': downsample_AtoB_warps.astype('float32'), 'BtoA_warps_128': downsample_BtoA_warps.astype('float32'),
                'AtoB_masks_128': downsample_AtoB_masks.astype('float32'), 'BtoA_masks_128': downsample_BtoA_masks.astype('float32'),
                'A_parsing_128': downsample_A_parsing.astype('float32'), 'B_parsing_128': downsample_B_parsing.astype('float32'),
                
                'face_A_warp': face_A_warp.astype('float32'), 'face_B_warp': face_B_warp.astype('float32'),
                'A_paths': A_path, 'B_paths': B_path,
                }

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'PoseDataset'
