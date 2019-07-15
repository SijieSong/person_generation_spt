import torch
import torch.nn as nn

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
def normalization_model(mean, std, gpu_ids=[]):
    #mean = torch.tensor(mean).to(gpu_ids[0])
    #std = torch.tensor(std).to(gpu_ids[0])
    model = Normalization(mean, std)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
    model.to(gpu_ids[0])
    model = torch.nn.DataParallel(model, gpu_ids)   
    return model

class Normalization(nn.Module):
    def __init__(self, mean, std, requires_grad=False):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.

        self.mean = mean#torch.tensor(mean).view(-1, 1, 1)
        self.std = std #torch.tensor(std).view(-1, 1, 1)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, img):
        # normalize img
        mean = torch.tensor(self.mean).view(-1,1,1)
        mean = mean.cuda()
        std = torch.tensor(self.std).view(-1,1,1)
        std = std.cuda()
        #return (img - self.mean) / self.std
        return (img-mean)/std
        #return img
