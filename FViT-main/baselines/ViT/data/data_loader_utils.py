# data_loader_utils.py
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Compose, Resize, ToTensor
from PIL import Image
from baselines.ViT.data.imagenet import Imagenet_Segmentation


def get_imagenet_dataloader(imagenet_seg_path, batch_size=1, num_workers=0):
    normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    test_img_trans = Compose([
        Resize((224, 224)),
        ToTensor(),
        normalize,
    ])
    test_lbl_trans = Compose([
        Resize((224, 224), Image.NEAREST),
    ])
    ds = Imagenet_Segmentation(imagenet_seg_path, transform=test_img_trans, target_transform=test_lbl_trans)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)