from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from basicsr.utils.registry import DATASET_REGISTRY
from os import path as osp
from torch.utils import data as data


@DATASET_REGISTRY.register()
class FGCVDataset(Dataset):
    """自定义数据集"""

    def __init__(self, opt):
        self.opt=opt
        self.images_path = opt["images_path"]
        self.images_class = opt["images_class"]
        self.transform = opt["transform"]
        # normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )
        # data_transform = {
        # "train": transforms.Compose([
        #                             transforms.Resize((510, 510), Image.BILINEAR),
        #                             transforms.RandomCrop((cfg.INPUT.data_size, cfg.INPUT.data_size)),
        #                             transforms.RandomHorizontalFlip(),
        #                             transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
        #                             transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
        #                             transforms.ToTensor(),
        #                             normalize]),

        # "val": transforms.Compose([transforms.Resize((510, 510), Image.BILINEAR),
        #                            transforms.CenterCrop((cfg.INPUT.data_size, cfg.INPUT.data_size)),
        #                            transforms.ToTensor(),
        #                            normalize])}
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        # print(str(self.images_path[item]).replace("._",""))#解决地址中出现 ._ 的错误地址
        # img = Image.open(self.images_path[item])
        img = Image.open(str(self.images_path[item]).replace("._", ""))
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            img = img.convert("RGB")
            # raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
