import torch
import torch.nn as nn

def gram_matrix(input):
    a, b, c, d = input.size()  
    # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a, b, c * d)  # resise F_XL into \hat F_XL

    G = torch.bmm(features, features.transpose(2,1))  # compute the gram product: B x C x C

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(c * d)


def patch_gram_matrix(input, pose_map):
	# feat_map: batchsize x pose_joint_num x H x W
	# input: batchsize x C x H x W
	batchsize, no_joint, c, d = pose_map.size() # conditional map (e.g., semantic map)
	a, b, c, d = input.size() # batchsize x C x H x W

	patch_gram = []
	for i in xrange(8):
		pose_map_ = pose_map[:,i] # batchsize x H x W
		pose_map_ = pose_map_.view(batchsize, 1, c * d) # batchsize, H x W

		masked_input = input.view(a, b, c * d)

		masked_input = pose_map_ * masked_input

		masked_input = masked_input.view(a,b,c,d)

		G = gram_matrix(masked_input) # batchsize x C x C

		patch_gram.append(G)

	output = torch.cat([_.unsqueeze(1) for _ in patch_gram], 1).contiguous()

	return output # batchsize x pose_joint_num x C x C

if __name__ == '__main__':
	# input = torch.Tensor((4,3,5,5))
	input = torch.ones((2,128,128,128))
	pose_map = torch.ones((2,18,128,128))
	output = patch_gram_matrix(input, pose_map)

