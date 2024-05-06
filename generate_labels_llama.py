from transformers import AutoTokenizer
import transformers
import torch
import json
import os
import random
import argparse


def generate_prompt(caption: str, category_set: list, samples: list) -> str:
    
    cap1, label1 = random.choice(samples)
    cap2, label2 = random.choice(samples)
    cap3, label3 = random.choice(samples)

    categories = ",".join(category_set)
    prompt = f"""Classify the following image description into one of the following list of classes: {category_set}. Provide only the class name without any additional text.
    Image description: {cap1}
    Category from the list: {label1}
    Image description: {cap2}
    Category from the list: {label2}
    Image description: {caption}
    Category from the list:
    """
    
#     prompt = f"""Classify the following image description into one of the following list of classes: {category_set}. Provide only the class name without any additional text.
#     Image description: {caption}
#     Category from the list:
#     """
    
    return prompt

def process_meta(meta: dict) -> str:
    title = meta["caption"]
    desc = meta["description"]
    tags = meta["tags"]

    ## clip title, desc and tags to 50 words each.
    title = " ".join(title.split(" ")[:50])
    desc = " ".join(desc.split(" ")[:50])
    tags = ",".join(tags.split(",")[:50])
    
    caption = "Tags: " + tags + "\n" + "Title: " + title + "\n" + "Description: " + desc + "\n"

    return caption

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='geoimnet', help='dataset to generate captions for')
parser.add_argument('--domain', type=str, default='usa', help='domain to generate captions for')
parser.add_argument('--source', type=str, default='asia', help='source domain to sample captions from')
parser.add_argument('--counter', type=int, default=None, help='what part of data to generate from')
args = parser.parse_args()

# model = "meta-llama/Llama-2-7b-chat-hf"
model = "/longtail-ssl/models/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93/"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

domainnet = json.load(open(f"metadata/{args.dataset}.json"))
category_set = [c["category_name"] for c in domainnet["categories"]]
## expand comma seperated categories into bigger list

category_to_id = {c["category_name"]:c["category_id"] for c in domainnet["categories"]}

if args.dataset == "geoyfcc":
    category_set_expanded = []
    for cset in category_set:
        all_cats = cset.split(",")
        category_set_expanded.extend(all_cats)
    category_set_expanded = [c.lstrip().rstrip() for c in category_set_expanded]
    category_to_id_expanded = {}
    for cset, cid in category_to_id.items():
        all_cats = cset.split(",")
        for c in all_cats:
            category_to_id_expanded[c] = cid
    category_to_id_expanded = {c.lstrip().rstrip():cid for c, cid in category_to_id_expanded.items()}
    category_set = category_set_expanded
    category_to_id = category_to_id_expanded

if args.dataset == "domainnet":
    dataset = domainnet[f"{args.source}_train"]
    id_to_caption = {c["image_id"]:c["blip2_cap"] for c in dataset["metadata"]}
    id_to_label = {c["image_id"]:c["class_name"] for c in dataset["annotations"]}

    samples = [(id_to_caption[ii], id_to_label[ii]) for ii in id_to_caption.keys()]

    dataset = domainnet[f"{args.domain}_train"]
    id_to_caption = {c["image_id"]:c["blip2_cap"] for c in dataset["metadata"]}
    id_to_label = {c["image_id"]:c["class_name"] for c in dataset["annotations"]}

    context_len = 2048

elif "geo" in args.dataset:
    dataset = domainnet[f"{args.source}_train"]
    id_to_caption = {c["image_id"]:process_meta(c) for c in dataset["metadata"]}
    id_to_label = {c["image_id"]:c["class_name"] for c in dataset["annotations"]}

    samples = [(id_to_caption[ii], id_to_label[ii]) for ii in id_to_caption.keys()]

    dataset = domainnet[f"{args.domain}_train"]
    id_to_caption = {c["image_id"]:process_meta(c) for c in dataset["metadata"]}
    id_to_label = {c["image_id"]:c["class_name"] for c in dataset["annotations"]}
    context_len = 4096

elif "ego" in args.dataset:
    dataset = domainnet[f"{args.source}_train"]
    id_to_caption = {c["segment_id"]:c['text_caption'] for c in dataset["descriptions"]}
    id_to_label = {c["segment_id"]:c["class_name"] for c in dataset["annotations"]}

    samples = [(id_to_caption[ii], id_to_label[ii]) for ii in id_to_caption.keys()]

    dataset = domainnet[f"{args.domain}_train"]
    id_to_caption = {c["segment_id"]:c['text_caption'] for c in dataset["descriptions"]}
    id_to_label = {c["segment_id"]:c["class_name"] for c in dataset["annotations"]}
    context_len = 2048

if not os.path.exists("llm_labels"):
    os.makedirs("llm_labels")
save_name = "llm_labels/llm_label_{}_{}_{}_7b.json".format(args.dataset, args.source, args.domain)

if os.path.exists(save_name):
    all_labels = json.load(open(save_name))["labels"]
    generated_ids = [int(cap["image_id"]) for cap in all_labels]
else:
    all_labels = []
    generated_ids = []

all_labels = []

for idx, cap in id_to_caption.items():

    if idx in generated_ids:
        continue
    
    prompt = generate_prompt(cap, category_set, samples)
    sequence = pipeline(prompt.strip(),
                        do_sample=True,
                        top_k=10,
                        num_return_sequences=1,
                        eos_token_id=tokenizer.eos_token_id,
                        max_length=context_len,
                        return_full_text=False)
    predicted = sequence[0]['generated_text'].strip().split("\n")[0]
    all_labels.append({
        'image_id': idx,
        'label_name': predicted,
        'label': category_to_id.get(predicted, -1)
    })
    
    if (len(all_labels)+1) % 1000 == 0:
        print("{}/{}".format(len(all_labels), len(id_to_caption)), flush=True)
        with open(save_name, "w") as fh:
            json.dump({"labels":all_labels}, fh, indent=4)
        
with open(save_name, "w") as fh:
    json.dump({"labels":all_labels}, fh, indent=4)

print("Done!!")