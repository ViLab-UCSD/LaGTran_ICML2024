import os
import torch.utils.data as data
from PIL import Image
import json
import torch
from typing import List, Dict, Any

_VALID_DOMAIN = ["ego", "exo"]
_VALID_SPLIT = ["train", "test", "val"]
_DEFAULT_STR = "NULL"


def default_loader(path):
    feat = torch.load(path)
    return feat

def default_ann(segid):
    return {
        "segment_id":segid,
        "category": int(0),
        "class_name": _DEFAULT_STR
    }

def default_text(segid):
    return {
        "segment_id":segid,
        "text_caption": _DEFAULT_STR,
    }

def default_meta(segid):
    return {
        "segment_id": segid,
        "start_time": _DEFAULT_STR,
        "end_time"  : _DEFAULT_STR,
    }


class Ego4dLoader(data.Dataset):

    def __init__(self, root_dir, json_path, domain, split="train", transform=None, target_transform=None,
                 loader=default_loader, return_ann=True, return_meta=False, return_text=False, 
                 _meta_keys=None, _text_keys=None):
        
        if split == "test":
            split = "val"

        if _text_keys is not None:
            assert isinstance(_text_keys, list), "text keys has to be a list."
            assert return_text
            _text_keys += list(default_text(0).keys())
        else:
            _text_keys = list(default_text(0).keys())
        _text_keys = list(set(_text_keys))
        
        if _meta_keys is not None:
            assert isinstance(_meta_keys, list), "meta keys has to be a list."
            assert return_meta
            _meta_keys += list(default_meta(0).keys())
        else:
            _meta_keys = list(default_meta(0).keys())
        _meta_keys = list(set(_meta_keys))

        if not isinstance(domain, list):
            domain = [domain]
        if not isinstance(split, list):
            split = [split]

        # assert all([d in _VALID_DOMAIN for d in domain]), "Invalid Domain".format(domain)
        assert all([s in _VALID_SPLIT for s in split]), "split has to be {}. {} not recognized".format("|".join(_VALID_SPLIT), split)

        self.root_dir = root_dir
        keytag = []
        for d in domain:
            for s in split:
                keytag.append("{}_{}".format(d, s))
        json_data = json.load(open(json_path))

        self.return_ann = return_ann
        self.return_text = return_text
        self.return_meta = return_meta

        self.info = json_data.get("info", None)
        self.category_mapping = json_data['categories']

        self.classname_to_id = {c["category_name"]:int(c["category_id"]) for c in self.category_mapping}
        self.id_to_classname = {v:k for k,v in self.classname_to_id.items()}
        
        clipdata = [json_data[kt] for kt in keytag]
        id_to_seg = {clip["id"]:clip for clipd in clipdata for clip in clipd["clips"]}

        id_to_ann = {seg_id:default_ann(seg_id) for seg_id in id_to_seg.keys()}
        id_to_text = {seg_id:default_text(seg_id) for seg_id in id_to_seg.keys()}
        id_to_meta = {seg_id:default_meta(seg_id) for seg_id in id_to_seg.keys()}

        if return_ann:
            id_to_ann = {ann["segment_id"]:ann for clipd in clipdata for ann in clipd["annotations"]}
            assert len(id_to_ann) >= len(id_to_seg), "Annotations Missing"

        if return_text:
            id_to_text = {text["segment_id"]:text for clipd in clipdata for text in clipd["descriptions"]}
            assert len(id_to_text) >= len(id_to_seg), "Locations Missing"

        if return_meta:
            id_to_meta = {meta["segment_id"]:meta for clipd in clipdata for meta in clipd["metadata"]}
            assert len(id_to_meta) >= len(id_to_seg), "Metadata Missing"

        ## combine clip, annotation and text
        self.data = []
        for segid in id_to_seg.keys():

            self.data.append((
                segid,
                id_to_seg[segid]["feature_file_name"],
                id_to_seg[segid]["feature_indices"],
                id_to_ann[segid]["category"],
                {k:id_to_text[segid].get(k,"NULL") for k in _text_keys},
                {k:id_to_meta[segid].get(k, "NULL") for k in _meta_keys}
            ))
        transform = lambda p:torch.mean(p, dim=0).squeeze()

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self.target = [id_to_ann[segid]["category"] for segid in id_to_seg.keys()] 
        self.also_target = [g[3] for g in self.data]

    def __getitem__(self, index):
        segid, file_path, feature_indices, target, text, metadata = self.data[index]
        text.pop("segment_id",None)
        metadata.pop("segment_id",None)

        file_path = os.path.join(self.root_dir, file_path)
        feature_matrix = self.loader(file_path) ## N,1,d
        index_tensor = torch.tensor(feature_indices, dtype=torch.long)
        feature_vector = feature_matrix[index_tensor] ## n,1,d

        if self.transform is not None:
            feature_vector = self.transform(feature_vector)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return_obj = [segid, feature_vector]

        if self.return_ann:
            return_obj.append(target)

        if self.return_text:
            return_obj.append(text)
        
        if self.return_meta:
            return_obj.append(metadata)

        return tuple(return_obj)

    def __len__(self):
        return len(self.data)
