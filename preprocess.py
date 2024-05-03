import os

import torch

CHECKPOINT_HOME = os.environ.get("CHECKPOINT_HOME")
os.makedirs(CHECKPOINT_HOME, exist_ok=True)
CUDA = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

import torchvision.datasets as dset
import torchvision.transforms as transforms

cap = dset.CocoCaptions(root='./datasets/example/tarinval',
                        annFile='./datasets/example/tarinval/captions_example_trainval2017.json',
                        transform=transforms.ToTensor())

print('Number of samples: ', len(cap))
img, target = cap[1]  # load 4th sample

print("Image Size: ", img.size())
print(target)
print(type(target))
print(type(img))


def encode_transform(ori_img, args):
    '''

    :param ori_img: ori_img 为 cv2 imread 对象
    :return: 10*10 PIL Image 对象
    '''

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    e_model = CHECKPOINT_HOME + "encoder_dict.pkl"
    encoder = Encoder(args)
    encoder.load_state_dict(torch.load(e_model))
    encoder.eval()
    if cuda:
        encoder.cuda()
    _trans = ori_img.ravel().reshape(1, 12288)
    _trans = torch.from_numpy(_trans)
    _trans = Variable(_trans.type(Tensor))
    _enc = encoder(_trans).reshape(10, 10)
    enc_img = parse_img(_enc.data)
    return enc_img
