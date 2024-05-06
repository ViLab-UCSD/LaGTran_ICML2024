import torchvision
from .randaugment import RandAugmentMC

# class ResizeImage():
#     def __init__(self, size):
#         if isinstance(size, int):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size
#     def __call__(self, img):
#         th, tw = self.size
#         return img.resize((th, tw))


# class PlaceCrop(object):

#     def __init__(self, size, start_x, start_y):
#         if isinstance(size, int):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size
#         self.start_x = start_x
#         self.start_y = start_y

#     def __call__(self, img):
#         th, tw = self.size
#         return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))

# def val_transform():
#     resize_size=256
#     crop_size=224
#     start_center = (resize_size - crop_size - 1) / 2
#     return torchvision.transforms.Compose([
#         ResizeImage(resize_size),
#         PlaceCrop(crop_size, start_center, start_center),
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# def train_transform():
#     resize_size=256
#     crop_size=224
#     return torchvision.transforms.Compose([ResizeImage(resize_size),
#                   torchvision.transforms.RandomResizedCrop(crop_size),
#                   torchvision.transforms.RandomHorizontalFlip(),
#                   torchvision.transforms.ToTensor(),
#                   torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def train_transform(img_crop_size=224):

    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.RandomResizedCrop(img_crop_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return trans

class TransformFixMatch(object):
    def __init__(self, crop_size=224):
        self.weak =torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.RandomResizedCrop(crop_size),
                torchvision.transforms.RandomHorizontalFlip(),
               
            ])

        self.strong = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.RandomResizedCrop(crop_size),
                torchvision.transforms.RandomHorizontalFlip(),
            RandAugmentMC(n=2, m=10)])
        
        self.normalize = torchvision.transforms.Compose([
             torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)

def val_transform(img_crop_size=224):

    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(img_crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return trans

def test_transform(img_crop_size=224):

    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(img_crop_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return trans

def transform(split, dual_aug=False, img_crop_size=224):

    trans = {
        "train" : train_transform(img_crop_size),
        "val" : val_transform(img_crop_size),
        "test" : test_transform(img_crop_size)
    }

    if dual_aug:
        trans["train"] = TransformFixMatch(img_crop_size)
    
    try:
        return trans[split]
    except:
        raise BaseException("Transforms for split {} not available, Choose [train/val/test]".format(split))
