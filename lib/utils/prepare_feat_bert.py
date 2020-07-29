import torch
from torch.autograd import Variable
from lib.pytorch_misc import enumerate_by_image
import numpy as np

def prepare_feat(feat, img_inds):
    num_im = img_inds[-1] + 1
    lengths = []
    for i, s, e in enumerate_by_image(img_inds):
        lengths.append(e - s)
    max_len = max(lengths)
    feat_output = []
    zero_tensor = torch.zeros([feat.size(-1)]).cuda(feat.get_device(),async=True)
    zero_tensor = Variable(zero_tensor)
    mask = np.zeros([num_im, 1, max_len, max_len])
    for i, s, e in enumerate_by_image(img_inds):
        feat_t = feat[s:e]
        num_objs = e - s
        mask[i,0,:num_objs,:num_objs] = 1
        if num_objs < max_len:
            num_repeat = max_len - num_objs
            zeros_exand = zero_tensor.repeat(num_repeat,1)
            feat_t = torch.cat([feat_t, zeros_exand],0)
        feat_output.append(feat_t[None,:,:])
    return torch.cat(feat_output), Variable(torch.from_numpy(mask.astype('float32')).cuda(feat.get_device(),async=True))

def postdiso_feat(feat, img_inds):

    feat_o = None
    for i, s, e in enumerate_by_image(img_inds):
        num_objs_i = e - s
        if feat_o is None:
            feat_o = feat[i,:num_objs_i]
        else:
            feat_o = torch.cat([feat_o, feat[i,:num_objs_i]])
    return feat_o