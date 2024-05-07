# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from https://github.com/facebookresearch/LaViLa/blob/main/lavila/data/datasets.py

import json
import numpy as np
import os.path as osp
import random

import decord
import torch

from typing import List, Dict, Any, Union


def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def video_loader(root, vid, second, end_second=None, chunk_len=300, fps=30, clip_length=32, jitter=False):
    if chunk_len == -1:
        vr = decord.VideoReader(osp.join(root, '{}'.format(vid)))
        second_offset = second
        if end_second is not None:
            end_second = min(end_second, len(vr) / vr.get_avg_fps())
        else:
            end_second = len(vr) / vr.get_avg_fps()
    else:
        chunk_start = int(second) // chunk_len * chunk_len
        second_offset = second - chunk_start
        vr = decord.VideoReader(osp.join(root, '{}'.format(vid), '{}.mp4'.format(chunk_start)))
    if fps == -1:
        fps = vr.get_avg_fps()

    # calculate frame_ids
    frame_offset = int(np.round(second_offset * fps))
    total_duration = max(int((end_second - second) * fps), clip_length)
    if chunk_len == -1:
        if end_second <= second:
            raise ValueError("end_second should be greater than second")
        else:
            frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length, jitter=jitter)
    else:
        frame_ids = get_frame_ids(frame_offset, frame_offset + total_duration, num_segments=clip_length, jitter=jitter)

    # load frames
    if max(frame_ids) < len(vr):
        try:
            frames = vr.get_batch(frame_ids).asnumpy()
        except decord.DECORDError as error:
            print(error)
            frames = vr.get_batch([0] * len(frame_ids)).asnumpy()
    else:
        # find the remaining frames in the next chunk
        try:
            frame_ids_part1 = list(filter(lambda frame_id: frame_id < len(vr), frame_ids))
            frames_part1 = vr.get_batch(frame_ids_part1).asnumpy()
            vr2 = decord.VideoReader(osp.join(root, '{}.mp4'.format(vid), '{}.mp4'.format(chunk_start + chunk_len)))
            frame_ids_part2 = list(filter(lambda frame_id: frame_id >= len(vr), frame_ids))
            frame_ids_part2 = [min(frame_id % len(vr), len(vr2) - 1) for frame_id in frame_ids_part2]
            frames_part2 = vr2.get_batch(frame_ids_part2).asnumpy()
            frames = np.concatenate([frames_part1, frames_part2], axis=0)
        # the next chunk does not exist; the current chunk is the last one
        except (RuntimeError, decord.DECORDError) as error:
            print(error)
            frame_ids = get_frame_ids(min(frame_offset, len(vr) - 1), len(vr), num_segments=clip_length, jitter=jitter)
            frames = vr.get_batch(frame_ids).asnumpy()

    frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
    return torch.stack(frames, dim=0)


def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    seg_size = float(end_frame - start_frame - 1) / num_segments
    seq = []
    for i in range(num_segments):
        start = int(np.round(seg_size * i) + start_frame)
        end = int(np.round(seg_size * (i + 1)) + start_frame)
        end = min(end, end_frame)
        if jitter:
            frame_id = np.random.randint(low=start, high=(end + 1))
        else:
            frame_id = (start + end) // 2
        seq.append(frame_id)
    return seq


def video_loader_by_frames(root, vid, frame_ids):
    vr = decord.VideoReader(osp.join(root, vid))
    try:
        frames = vr.get_batch(frame_ids).asnumpy()
        frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
    except (IndexError, decord.DECORDError) as error:
        print(error)
        print("Erroneous video: ", vid)
        frames = [torch.zeros((240, 320, 3)) for _ in range(len(frame_ids))]
    return torch.stack(frames, dim=0)


class VideoDatasetBase(torch.utils.data.Dataset):
    """
    Create video dataloader for classification.
    """
    def __init__(self, dataset, root, metadata, domain="ego", split="train"):
        self.dataset = dataset
        self.root = root
        self.domain = domain
        self.split = split

        if self.dataset != "ego_exoDA" or split not in ["train", "val"]:
            raise NotImplementedError

        ego4d = json.load(open(metadata))["{}_{}".format(self.domain, self.split)]
        self.samples = []
        for vid, ann, narration, meta in zip(ego4d["clips"], ego4d["annotations"], ego4d["descriptions"], ego4d["metadata"]):
            segment_id = vid["id"]
            if segment_id in [1908762465, 128702527, 3023277400]: ## zero-len problematic segments.
                continue
            # assert segment_id == ann["segment_id"] == narration["segment_id"] == meta["segment_id"]
            if not (segment_id == ann["segment_id"] == narration["segment_id"] == meta["segment_id"]):
                print("Mismatch in segment_id: ", segment_id, ann["segment_id"], narration["segment_id"], meta["segment_id"])
            vid_name = vid["video_file_name"]
            start_second = meta["start_time"] 
            end_second = meta["end_time"]
            label = ann["category"]
            narration = narration["text_caption"]
            self.samples.append((segment_id, vid_name, start_second, end_second, label, narration))


    def get_raw_item(self, i, is_training=True, num_clips=1, clip_length=32):
        """
        Irrespective of the length of the video, always sample <clip_len> equally spaced frames.
        Dense sampling for shorter videos but sparser sampling for longer videos.
        """
        segid, vid, start, end, label, narration = self.samples[i]
        frames = video_loader(self.root, vid, start,
                                end_second=end,
                                chunk_len=-1, 
                                fps=-1, 
                                clip_length=clip_length,
                                jitter=is_training)
        return segid, frames, label, narration

    def __getitem__(self, i):
        # override this method in the downstream dataset
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)


class VideoClassyDataset(VideoDatasetBase):
    def __init__(
        self, 
        dataset, 
        root: str, 
        metadata: str, 
        transform=None,
        label_mapping=None,
        num_clips: int=1,
        clip_length: int=8, 
        clip_stride: int=1,
        domain: str="ego", 
        split: str="train",
        use_extra: bool=True,
        return_narration: bool=False,
        **kwargs
    ):
        if split == "test":
            split = "val"
        super().__init__(dataset, root, metadata, domain, split, use_extra)

        self.transform = transform
        self.is_training = split == 'train' 
        self.label_mapping = label_mapping
        self.num_clips = num_clips
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.return_narration = return_narration

        self.target = [self.samples[i][4] for i in range(len(self.samples))]

    def __getitem__(self, i):
        segid, frames, label, narration = self.get_raw_item(
            i, is_training=self.is_training,
            num_clips=self.num_clips,
            clip_length=self.clip_length,
        )

        # apply transformation
        if self.transform is not None:
            frames = self.transform(frames)

        if self.label_mapping is not None:
            ego4d = json.load(open(self.metadata))["categories"]
            self.label_mapping = {c["category_id"]: c["category_name"] for c in ego4d}

        return_obj = (segid, frames, label)
        if self.return_narration:
            return_obj += (narration,)
        
        return return_obj

def get_downstream_dataset(transform, args, subset='train', label_mapping=None):
    return VideoClassyDataset(
        args.dataset,
        args.root,
        args.metadata_train if subset == 'train' else args.metadata_val,
        transform,
        is_training=(subset == 'train'),
        label_mapping=label_mapping,
        num_clips=args.num_clips,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        sparse_sample=args.sparse_sample,
    )
