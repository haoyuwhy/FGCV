from PIL import Image
import PIL
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from basicsr.utils.registry import DATASET_REGISTRY
from os import path as osp
from torch.utils import data as data
from basicsr.utils import scandir

@DATASET_REGISTRY.register()
class FGVCDataset(Dataset):
    """自定义数据集"""

    def __init__(self, opt):
        self.opt=opt
        self.images_path = opt["images_path"]
        if type(self.images_path)==str:
            self.paths = [osp.join( self.images_path, v)
                    for v in list(scandir( self.images_path))]
            self.images_path=self.paths
        if(opt['phase']!="test"):
            self.images_class = opt["images_class"]
        self.transform = opt["transform"]

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        # print(str(self.images_path[item]).replace("._",""))#解决地址中出现 ._ 的错误地址
        # img = Image.open(self.images_path[item])
        try:
            img = Image.open(str(self.images_path[item]).replace("._", ""))
            if(self.opt['phase']!="test"):
                label = self.images_class[item]
        except PIL.UnidentifiedImageError:
            img = Image.open(str(self.images_path[item-1]).replace("._", ""))
            if(self.opt['phase']!="test"):
                label = self.images_class[item-1]
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            img = img.convert("RGB")
            # raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        if self.transform is not None:
            img = self.transform(img)
        if(self.opt['phase']!="test"):
            return img, label,self.images_path[item]
        else:
            
            return [img.squeeze(0),osp.basename(self.images_path[item])]

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
