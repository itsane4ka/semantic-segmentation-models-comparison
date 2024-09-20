import torch
import torch.utils.data as data
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
from matplotlib import pyplot as plt
import numpy as np
import random


class OutdoorDataset(data.Dataset):
    def __init__(self, root, split='train'):
        self.dataset = 'OutdoorDataset'
        self.root = root
        self.split = split
        self.datapath = []  # every element contains path for images & path for mask
        with open(os.path.join(self.root, split) + '/' + split + '.txt', 'r') as f:
            ids = f.readlines()
        for id in ids:
            id = id.replace("\n", "")
            self.datapath.append(
                {
                    'name': id,
                    'img': os.path.join(self.root, split + '/images/' + id + '.png'),
                    'mask': os.path.join(self.root, split + '/labels/' + id + '.png')
                }
            )

    def transform(self, image, mask):
        # Resize
        resize_img = transforms.Resize(size=(800, 600), interpolation=Image.BILINEAR)
        resize_mask = transforms.Resize(size=(800, 600), interpolation=Image.NEAREST)
        image = resize_img(image)
        mask = resize_mask(mask)

        # Random Crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(512, 512))
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = F.vflip(image)
            mask = F.vflip(mask)

        # Transform to tensor
        image = F.to_tensor(image) * 255
        mask = F.to_tensor(mask) * 255

        return image, mask

    def __getitem__(self, index):
        img_path = self.datapath[index]['img']
        img = Image.open(img_path)
        mask_path = self.datapath[index]['mask']
        mask = Image.open(mask_path)

        if self.split == 'train':
            img, mask = self.transform(img, mask)
        else:
            img = F.to_tensor(img) * 255
            mask = F.to_tensor(mask) * 255

        return img, mask, self.datapath[index]['name']

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    dataset = OutdoorDataset('data/', split='val')
    img, msk, id = dataset[0]
    print(id)
    img_np = img.numpy() / 255
    img_np = np.transpose(img_np, [1, 2, 0])
    plt.imshow(img_np)
    plt.show()

    print(img.shape)
    print(msk.shape)
    msk_np = msk.numpy()
    msk_np = np.repeat(msk_np, 3, axis=0)
    print(msk_np.shape)
    msk_np = msk_np / 6
    msk_np = np.transpose(msk_np, [1, 2, 0])
    plt.imshow(msk_np)
    plt.show()
