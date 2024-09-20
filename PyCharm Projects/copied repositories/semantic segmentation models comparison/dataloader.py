from dataset import OutdoorDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_data_loader(root, batch_size, split='train', num_workers=4):
    dset = OutdoorDataset(root=root, split=split)
    loader = DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader


if __name__ == '__main__':
    trainloader = get_data_loader('data', batch_size=2, split='train')
    testloader = get_data_loader('data', batch_size=1, split='val')
    for iter, (img, msk, id) in enumerate(trainloader):
        pass
    for iter, (img, msk, id) in enumerate(testloader):
        pass
