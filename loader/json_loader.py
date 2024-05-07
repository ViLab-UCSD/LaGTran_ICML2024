import os
import torch.utils.data as data
from PIL import Image
import json
from typing import List, Dict, Any, Callable, Union

_VALID_SPLIT = ["train", "test"]
_DEFAULT_STR = "NULL"

def default_loader(path):
    return Image.open(path).convert('RGB')

def default_ann(imid):
    return {
        "image_id":imid,
        "category": int(0),
        "class_name": _DEFAULT_STR
    }

def default_meta(imid):
    return {
        "image_id":imid,
        "caption": _DEFAULT_STR,
        "tags"       : _DEFAULT_STR,
        "description"       : _DEFAULT_STR,
        "url"        : _DEFAULT_STR
    }

def default_loc(imid):
    return {
        "image_id": imid,
        "countryRegionIso2": _DEFAULT_STR,
        "continentRegion"  : _DEFAULT_STR,
        "latitude"         : 0.0,
        "longitude"        : 0.0
    }


class ImageJSONLoader(data.Dataset):

    def __init__(
            self,
            root_dir: str,
            json_path: str,
            domain: Union[str, List[str]],
            split: str = "train",
            transform=None,
            target_transform=None,
            loader=default_loader,
            return_ann: bool=True,
            return_loc: bool=False,
            return_meta: bool=False,
            _loc_keys: List[str]=None,
            _meta_keys: List[str]=None
    ):

        if _loc_keys is not None:
            if not isinstance(_loc_keys, list):
                raise ValueError("loc keys has to be a list.")
            if not return_loc:
                raise ValueError("return_loc has to be True to use loc keys.")
            _loc_keys += list(default_loc(0).keys())
        else:
            _loc_keys = list(default_loc(0).keys())
        _loc_keys = list(set(_loc_keys))
        
        if _meta_keys is not None:
            if not isinstance(_meta_keys, list):
                raise ValueError("meta keys has to be a list.")
            if not return_meta:
                raise ValueError("return_meta has to be True to use meta keys.")
            _meta_keys += list(default_meta(0).keys())
        else:
            _meta_keys = list(default_meta(0).keys())
        _meta_keys = list(set(_meta_keys))

        if not isinstance(domain, list):
            domain = [domain]
        if not isinstance(split, list):
            split = [split]

        if not all([s in _VALID_SPLIT for s in split]):
            raise ValueError("split has to be {}. {} not recognized".format("|".join(_VALID_SPLIT), split))

        self.root_dir = root_dir
        keytag = []
        for d in domain:
            for s in split:
                keytag.append("{}_{}".format(d, s))
        json_data = json.load(open(json_path))

        self.return_ann = return_ann
        self.return_loc = return_loc
        self.return_meta = return_meta

        self.info = json_data.get("info", None)
        self.category_mapping = json_data['categories']

        self.classname_to_id = {c["category_name"]:int(c["category_id"]) for c in self.category_mapping}
        self.id_to_classname = {v:k for k,v in self.classname_to_id.items()}
        
        imdata = [json_data[kt] for kt in keytag]
        id_to_im = {im["id"]:im for imd in imdata for im in imd["images"]}

        id_to_ann = {image_id:default_ann(image_id) for image_id in id_to_im.keys()}
        id_to_loc = {image_id:default_loc(image_id) for image_id in id_to_im.keys()}
        id_to_meta = {image_id:default_meta(image_id) for image_id in id_to_im.keys()}

        if return_ann:
            id_to_ann = {ann["image_id"]:ann for imd in imdata for ann in imd["annotations"]}
            assert len(id_to_ann) >= len(id_to_im), "Annotations Missing"

        if return_loc:
            id_to_loc = {loc["image_id"]:loc for imd in imdata for loc in imd["locations"]}
            assert len(id_to_loc) >= len(id_to_im), "Locations Missing"

        if return_meta:
            id_to_meta = {meta["image_id"]:meta for imd in imdata for meta in imd["metadata"]}
            assert len(id_to_meta) >= len(id_to_im), "Metadata Missing"

        ## combine image, annotation and locations
        self.data = []
        for imid in id_to_im.keys():

            self.data.append((
                imid,
                id_to_im[imid]["filename"],
                id_to_ann[imid]["category"],
                {k:id_to_loc[imid].get(k,"NULL") for k in _loc_keys},
                {k:id_to_meta[imid].get(k, "NULL") for k in _meta_keys}
            ))

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self.target = [g[2] for g in self.data]

    def __getitem__(self, index):
        imid, impath, target, location, metadata = self.data[index]
        location.pop("image_id",None)
        metadata.pop("image_id",None)

        impath = os.path.join(self.root_dir, impath)
        img = self.loader(impath)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return_obj = [imid, img]

        if self.return_ann:
            return_obj.append(target)

        if self.return_loc:
            return_obj.append(location)
        
        if self.return_meta:
            metadata["combined"] = " ".join([metadata["caption"], metadata["tags"].replace(","," "), metadata["description"]])
            return_obj.append(metadata)

        return tuple(return_obj)


    def __len__(self):
        return len(self.data)

