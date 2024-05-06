from transformers import AutoTokenizer
# from datasets import Dataset
# from transformers.pipelines.pt_utils import KeyDataset
import transformers
import torch
import json
import os
import random
import argparse
import pandas as pd    

# from tqdm.notebook import tqdm
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("source_domain", type=str)
parser.add_argument("target_domain", type=str)
parser.add_argument("--model", type=str, default="/home/tarun/.cache/huggingface/hub/")
parser.add_argument("--is_quantized", action="store_true")
args = parser.parse_args()

source, target = args.source_domain, args.target_domain
target_data = json.load(open("metadata/domainnet.json"))[f"{target}_train"]['metadata']

# model = "/home/tarun/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/1448453bdb895762499deb4176c1dd83b145fac1"
model = os.path.join(args.model, "models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/1448453bdb895762499deb4176c1dd83b145fac1")

if args.is_quantized:
    model_kwargs = {
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True},
        "low_cpu_mem_usage": True,
    
    }
else:
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
    }

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    device_map="auto",
    model_kwargs=model_kwargs
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


all_outs = ""
for idx, meta in enumerate(tqdm(target_data), 1):

    fid = meta["image_id"]
    blip2_cap = meta["blip2_cap"]

    messages = [
    {"role": "system", "content": f"You are an AI expert at transforming the style of captions of {target} images to look as if they were captured in {source} domain. Remember to just change the style of the caption, not the objects mentioned in the caption. Make as minimum changes to the sentence as possible. "},
    {"role": "user", "content": f"Original sentence: A {target} image of a X.\n Transformed sentence: A {source} image of a X.\n\nOriginal sentence:{blip2_cap}.\nTransformed sentence: "},
    # {"role": "user", "content": f"Original sentence in {target} domain:{blip2_cap}.\nTransformed sentence in {source} domain: "},
]   
    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    outputs = pipeline(
        prompt,
        max_new_tokens=96,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    out = outputs[0]["generated_text"][len(prompt):].lower().lstrip().rstrip()
    # all_outs.append({
    #     "fid": fid,
    #     "original_sentence": blip2_cap,
    #     f"transformed_{source}": out
    # })
    all_outs += f"{fid}\t{blip2_cap}\t{out}\n"

    ## create the dir
    if not os.path.exists(f"metadata/llm_adapted_captions/"):
        os.makedirs(f"metadata/llm_adapted_captions/", exist_ok=True)

    if idx % 100 == 0:
        with open(f"metadata/llm_adapted_captions/{source}_{target}.csv", "w") as f:
            f.write(all_outs)