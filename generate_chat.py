from transformers import AutoTokenizer
# from datasets import Dataset
# from transformers.pipelines.pt_utils import KeyDataset
import transformers
import torch
import json
import os
import random
import argparse
# from tqdm.notebook import tqdm
from tqdm import tqdm

# os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3,5,6"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='geoimnet', help='dataset to generate captions for')
parser.add_argument('--domain', type=str, default='asia', help='domain to generate captions for')
args = parser.parse_args()

def strip_prefix(caption):
    
    if caption.startswith("An image of a "):
        caption = caption.split("An image of a ")[-1]
    elif caption.startswith("An image of "):
        caption = caption.split("An image of ")[-1]
    elif caption.startswith("An image of... "):
        caption = caption.split("An image of... ")[-1]

    return caption

metadata_path = {
    "geoimnet" : "geoimnet.json",
    "geoplaces" : "geoplaces.json",
    "geoyfcc" : "geoyfcc.json"
}

geoimnet = json.load(open("metadata/{}".format(metadata_path[args.dataset])))["{}_train".format(args.domain)]
# geoimnet = json.load(open("{}".format(metadata_path[args.dataset])))["{}_train".format(args.domain)]
id_to_cat = {ann["image_id"]:ann["class_name"] for ann in geoimnet["annotations"]}

name = {
    "13b" : "13f8d72c0456c17e41b3d8b4327259125cd0defa",
    "70b" : "cfe96d938c52db7c6d936f99370c0801b24233c4"
}

size = "13b"

# model = "/newdata/tarun/models/llama/models--meta-llama--Llama-2-{}-chat-hf/snapshots/{}".format(size, name[size]) #"meta-llama/Llama-2-13b-chat-hf"
model = "/longtail-ssl/models/models--meta-llama--Llama-2-{}-chat-hf/snapshots/{}".format(size, name[size])

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

system_prompt_summary = """\
Below is a brief caption and comma separated tags delimited by triple backticks extracted from an image. Translate each of them into English first if necessary, and then extract what could be a condensed caption for the image based on this data. Follow these instructions:
0. Only focus on aspects that convey what is in the image and ignore all the noise.
1. Keep your answer to a MAXIMUM of 5-10 words and answer in single sentence. 
2. Start your answer with "An image of ...", followed by the condensed caption, and put NOTHING ELSE.
3. DON'T put geographical info, like country and city names, in the answer. Also ignore additional details like proper nouns, camera models, date and time in the answer."""

# shorter_prompt = """\
# Generate a concise 5-10 word image caption in English using the title, description and comma separated tags provided. Start your answer with "An image of ...", followed by the caption, but NOTHING ELSE. Exclude geography and specific details, and emphasize the image's core content."""

# system_prompt_classname = """\
# Below is the title, description and comma separated tags delimited by triple backticks extracted from an image. Translate each of them into English first if necessary, and then extract what could be the possible category of the image from the IMAGENET dataset with 1000 classes. Only output the possible class name in the answer and NOTHING else. NO explanation necessary."""

# system_prompt_classname = """\
# Below is the title, description and comma separated tags delimited by triple backticks extracted from an image. Translate each of them into English first if necessary, and then extract what could be the possible category of the image from the Places-205 dataset with 205 classes. Only output the possible class name, in English, in the answer and NOTHING else. NO explanation necessary."""

all_metadata = geoimnet["metadata"]
# random.shuffle(all_metadata)
save_name = "metadata/caponly_condensed_caption_{}_{}_13b.json".format(args.dataset, args.domain)

if os.path.exists(save_name):
    all_generated_captions = json.load(open(save_name))["captions"]
    generated_flickrids = [int(cap["flickr_id"]) for cap in all_generated_captions]
else:
    all_generated_captions = []
    generated_flickrids = []

# all_prompts = []
for meta in tqdm(all_metadata):
    flickr_id = meta["image_id"]
    title = meta["caption"]
#     desc = meta["description"]
    tags = meta["tags"]
#     blip_cap = meta["blip2_cap"]

    if int(flickr_id) in generated_flickrids:
        continue
    if len(title.split(" ")) > 120:
        title = " ".join(title.split(" ")[:100])
#     if len(desc.split(" ")) > 120:
#         desc = " ".join(desc.split(" ")[:100])
    if len(tags.split(",")) > 120:
        tags = ",".join(tags.split(",")[:100])
#     if len(blip_cap.split(" ")) > 15:
#         blip_cap = " ".join(blip_cap.split(" ")[:15])
    
    prompt = f"""\
        <s>[INST] 
        {system_prompt_summary}

        ```
        Caption: {title}
        Tags: {tags}
        ```
        [/INST]
        """

    tokens = tokenizer.tokenize(prompt)
    # if len(tokens) > 2048:
    #     import pdb; pdb.set_trace()

    sequences = pipeline(
        f"{prompt}\n",
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=2048,
        return_full_text=False)

    generated = sequences[0]['generated_text'].strip()
    generated = strip_prefix(generated)
    # import pdb; pdb.set_trace()
    
    all_generated_captions.append({
        "flickr_id" : flickr_id,
        "condensed_caption" : generated,
        "gt_category" : id_to_cat[flickr_id],
    })

    if (len(all_generated_captions)-10) % 1000 == 0:
        print("{}/{}".format(len(all_generated_captions), len(all_metadata)), flush=True)
        with open(save_name, "w") as fh:
            json.dump({"captions":all_generated_captions}, fh, indent=4)
        
with open(save_name, "w") as fh:
    json.dump({"captions":all_generated_captions}, fh, indent=4)

print("Done!!")