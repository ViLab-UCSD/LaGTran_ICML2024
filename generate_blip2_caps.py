import json
from transformers import Blip2Processor, Blip2ForConditionalGeneration

import torch
import torch.utils.data as data
from tqdm import tqdm

from loader.json_loader import ImageJSONLoader

path = "/home/tarun/.cache/huggingface/hub/models--Salesforce--blip2-opt-2.7b/snapshots/6e723d92ee91ebcee4ba74d7017632f11ff4217b"

processor = Blip2Processor.from_pretrained(path)
model = Blip2ForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float16)
model = model.cuda()

# names = ["GeoImnet"]
# targets = ["asia", "usa"]

names = ["domainnet"]
domains = ["real", "clipart", "painting", "sketch", "quickdraw", "infograph"]
# domains = ["quickdraw", "infograph"]

for name in names:
    for target in domains:
        # root_dir = f"/home/tarun/metadata/{name}/"
        root_dir="/newfoundland/tarun/datasets/Adaptation/visDA/"
        # root_dir=""

        dataset = ImageJSONLoader(root_dir=root_dir, 
                                    json_path=f"metadata/{name.lower()}.json",
                                    transform=processor.image_processor,
                                    domain=target,
                                    split="test")
        train = data.DataLoader(dataset, batch_size=32,  shuffle=False,
                                drop_last=False, pin_memory=False, num_workers=4)

        captions = []
        # print(len(train))
        # print("blip2_captions_{}_{}.json".format(name.lower(), target))

        for batch in tqdm(train):
            
            fid = batch[0]
            images = {k:v[0].cuda() for k,v in batch[1].items()}
            # text_prompt = processor(images=None, text=["Generate a short caption ignoring the style and only focusing on the content."]*len(fid), return_tensors="pt").to("cuda")
            # images.update(text_prompt)
            generated_ids = model.generate(**images, max_new_tokens=25)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            captions.extend([{"flickr_id":f, "blip_caption":s.strip()} for f,s in zip(fid.tolist(), generated_text)])
            

        with open("metadata/domainnet_blip/blip2_captions_{}_{}_test.json".format(name.lower(), target), "w") as fh:
            json.dump({"captions":captions}, fh, indent=4)

