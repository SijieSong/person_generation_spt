"""
Modified VGG16 to compute perceptual loss.
This class is mostly copied from pytorch/examples.
See, fast_neural_style in https://github.com/pytorch/examples.
"""

import torch
from torchvision import models
import numpy


class VGG_OUTPUT(object):

    def __init__(self, relu1_2, relu2_1):
        self.__dict__ = locals()


class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        model = models.vgg16(pretrained=True)
        model.load_state_dict(model_path)
        vgg_pretrained_features = model.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_1 = h

        return h_relu2_1

def vgg_model(gpu_ids=[]):
    model = VGG16()
    if len(gpu_ids) > 0:
	assert(torch.cuda.is_available())
	model.to(gpu_ids[0])
	model = torch.nn.DataParallel(model, gpu_ids)
    return model

if __name__ == '__main__':
    #a = VGG16()
    b = numpy.random.randint(0,10,size=(1,3,256,256))
    b = torch.Tensor(b)
    a.forward(b)


