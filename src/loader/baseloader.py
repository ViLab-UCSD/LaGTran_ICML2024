import os
import logging

from loader.utils import default_loader
from loader.transforms import transform
from loader.video_transforms import video_transform
from loader.sampler import get_sampler
import torch.utils.data as data


logger = logging.getLogger('mylogger')


class BaseLoader():
    """Function to build data loader(s) for the specified splits given the parameters.
    """

    def __init__(self, cfg, splits, batch_size):
        data_root = cfg.get('data_root', '/path/to/dataset')
        if not os.path.isdir(data_root):
            raise Exception('{} does not exist'.format(data_root))

        num_workers = cfg.get('n_workers', 4)
        smplr_dict = cfg.get('sampler', {'name': 'random'})

        self.data_loader = dict()
        for split in splits:

            json_file = cfg.get("json_file", None)
            if not os.path.isfile(json_file):
                raise Exception('{} not available'.format(json_file))
            
            if "Video" in cfg["loader"]:
                trans = video_transform(split)
            else:
                trans = transform(split)
            drop_last = cfg.get('drop_last', False) if 'train' in split else False
            shuffle = split == 'train'

            if "loc" in cfg:
                kwargs = {
                    "domain" : cfg["domain"],
                    "return_ann" : cfg.get("ann", "False"),
                    "return_loc" : cfg.get("loc", "False"),
                    "return_meta" : cfg.get("meta", "False"),
                }
            elif "text" in cfg:
                kwargs = {
                    "domain" : cfg["domain"],
                    "return_ann" : cfg.get("ann", "False"),
                    "return_text" : cfg.get("text", "False"),
                    "return_meta" : cfg.get("meta", "False")
                }
            else:
                kwargs = {
                    "domain" : cfg["domain"],
                    "clip_length" : cfg.get("clip_length", 8),
                    "dataset"     : cfg.get("dataset", "ego_exoDA"),
                }

            dataset = self.getDataset(root_dir=data_root, json_file=json_file, split=split, transform=trans, loader=default_loader, **kwargs)

            if ('train' in split) and (smplr_dict['name'] != 'random'):
                sampler = get_sampler(dataset, smplr_dict)
                self.data_loader[split] = data.DataLoader(
                    dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
                    drop_last=drop_last, pin_memory=True, num_workers=num_workers
                )
            else:
                self.data_loader[split] = data.DataLoader(
                    dataset, batch_size=batch_size,  sampler=None, shuffle=shuffle,
                    drop_last=drop_last, pin_memory=False, num_workers=num_workers
                )

            logger.info("{split}: {size}".format(split=split, size=len(dataset)))

    def getDataset(self):

        raise NotImplementedError