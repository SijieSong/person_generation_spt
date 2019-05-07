import os
import numpy
from options.test_pose_options import TestPoseOptions
from data import CreateDataLoader
from models import create_model
from skimage.io import imread, imsave
from scipy.misc import imresize

def deprocess_image(img):
        return (255 * ((img + 1) / 2.0)).astype(numpy.uint8)

if __name__ == '__main__':
    opt = TestPoseOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  ##### randomly choose the target pose ####
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display

    opt.no_lsgan= False


    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)
   
    if not os.path.exists(opt.results_dir):
        os.makedirs(opt.results_dir)

    # test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
	
        model.test()

        fake_B = model.fake_B.cpu().numpy()
        fake_B = fake_B[0]
        fake_B = numpy.transpose(fake_B,(1,2,0))
        
        fake_A = model.fake_A.cpu().numpy()
        fake_A = fake_A[0]
        fake_A = numpy.transpose(fake_A,(1,2,0))
        img_A_path = model.image_A_paths[0]
        img_B_path = model.image_B_paths[0]    
	

        A_parsing = model.A_parsing.cpu().numpy()

        A = imread(img_A_path)
        B = imread(img_B_path)


        fake_A = deprocess_image(fake_A)
        fake_B = deprocess_image(fake_B)

        A_name = img_A_path.split('/')[-1]
        B_name = img_B_path.split('/')[-1]

        imname = os.path.join(opt.results_dir, A_name + '_' + B_name +'.png')
    
        imsave(imname,numpy.concatenate((A, B, fake_B),axis=1))

