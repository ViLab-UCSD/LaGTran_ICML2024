import torchvision

def train_transform(img_crop_size=224):

    trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.RandomResizedCrop(img_crop_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    return trans

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

def transform(split, img_crop_size=224):

    if split not in ["train", "val", "test"]:
        raise BaseException("Transforms for split {} not available, Choose [train/val/test]".format(split))

    trans = {
        "train" : train_transform(img_crop_size),
        "val" : val_transform(img_crop_size),
        "test" : test_transform(img_crop_size)
    }

    return trans[split]
