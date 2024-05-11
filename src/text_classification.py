from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler 
import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
import os
import argparse

from loader.transforms import val_transform
from loader.json_loader import ImageJSONLoader
from loader.ego4d_loader import Ego4dLoader
from loader.sampler import BalancedSampler

## Sample command: python3 text_classification_distilbert.py --dataset GeoPlaces --source asia --target usa --n_iters 50000 --batch_size 32 --learning_rate 5e-5 --root_dir data

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="GeoPlaces", choices=["GeoPlaces", "GeoImnet", "GeoUniDA", "DomainNet","EgoExoDA"])
parser.add_argument("--source", type=str)
parser.add_argument("--target", type=str)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--root_dir", type=str, default="data")
parser.add_argument("--n_iters", type=int, default=15000)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--backbone", type=str, default="distilbert-base-uncased")
parser.add_argument("--wd", type=float, default=3e-4)

args = parser.parse_args()

if args.dataset == "GeoPlaces":
    N_CLASSES = 205
    cap_src = "combined_captions"
elif args.dataset == "GeoImnet":
    N_CLASSES = 600
    cap_src = "combined_captions"
elif args.dataset.lower() == "GeoUniDA".lower():
    N_CLASSES = 62
    cap_src = "combined_captions"
elif args.dataset.lower() == "DomainNet".lower():
    N_CLASSES = 345
    cap_src = "blip2_cap"
elif args.dataset.lower() == "Ego2Exo".lower():
    N_CLASSES = 24
    cap_src = "text_caption"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def lower(s): return s.lower()

BATCH_SIZE = args.batch_size
n_iters = args.n_iters
name = args.dataset
source = args.source
target = args.target
exp_name="{}_{}_{}_captions_{}".format(name, source, target, cap_src)

print("Saving to {}".format(exp_name))

tokenizer = AutoTokenizer.from_pretrained(args.backbone, model_max_length=256)
model = AutoModelForSequenceClassification.from_pretrained(args.backbone, num_labels=N_CLASSES).cuda()
if args.backbone == "gpt2":
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

model.train()

if name in ["GeoPlaces", "GeoImnet", "GeoUniDA", "DomainNet"]:
    train_dataset = ImageJSONLoader(root_dir=args.root_dir, 
                        json_path=f"metadata/{name.lower()}.json",
                        transform=val_transform(256),
                        domain=source,
                        split="train",
                        return_meta=True,
                        _meta_keys=[cap_src])
else:
    train_dataset = Ego4dLoader(root_dir=args.root_dir, 
                        json_path=f"metadata/{name.lower()}.json",
                        domain=source,
                        split="train",
                        return_text=True,
                        _text_keys=[cap_src])

sampler = BalancedSampler(train_dataset)
shuffle = False

train = data.DataLoader(train_dataset, batch_size=BATCH_SIZE,  shuffle=shuffle, sampler=sampler,
                    drop_last=True, pin_memory=False, num_workers=4)

loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

params = model.parameters()
optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.wd)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=n_iters
)

if not os.path.exists(os.path.join("bert_checkpoints", exp_name)):
    os.makedirs(os.path.join("bert_checkpoints", exp_name))

best_acc = 0
iter_ds = iter(train)
for it in range(n_iters):
    model.train()
    try:
        _, _, label, text = next(iter_ds)
    except:
        iter_ds = iter(train)
        _, _, label, text = next(iter_ds)
         
    optimizer.zero_grad()

    tokens = tokenizer(text[cap_src], padding="longest", truncation=True, return_tensors="pt")
    tokens.update({"labels":label})
    tokens = {k:v.cuda() for k,v in tokens.items()}
    outputs = model(**tokens)

    loss = outputs.loss

    loss.backward()
    optimizer.step()
    lr_scheduler.step()

torch.save(model.state_dict(), "bert_checkpoints/{}/best.pth".format(exp_name))

## compute target preditions using the best model.

if name in ["GeoPlaces", "GeoImnet", "GeoUniDA", "DomainNet"]:
    target_dataset = ImageJSONLoader(root_dir=args.root_dir, 
                        json_path=f"metadata/{name.lower()}.json",
                        transform=val_transform(256),
                        domain=target,
                        split="train",
                        return_meta=True,
                        _meta_keys=[cap_src])
else:
    target_dataset = Ego4dLoader(root_dir=args.root_dir, 
                        json_path=f"metadata/{name.lower()}.json",
                        domain=target,
                        split="train",
                        return_text=True,
                        _text_keys=[cap_src])

target_train = data.DataLoader(target_dataset, batch_size=BATCH_SIZE,  shuffle=False, drop_last=False, num_workers=4)

print("\nComputing target predictions...")
model = AutoModelForSequenceClassification.from_pretrained(args.backbone, num_labels=N_CLASSES).cuda()
model.load_state_dict(torch.load("bert_checkpoints/{}/best.pth".format(exp_name)))
model.eval()

all_preds = []
all_labels = []
all_ids = []

for image_id, _, label, text in tqdm(target_train):
         
    tokens = tokenizer(text[cap_src], padding="longest", truncation=True, return_tensors="pt")
    tokens = {k:v.cuda() for k,v in tokens.items()}
    with torch.no_grad():
        outputs = model(**tokens)

    prediction = outputs.logits.detach()
    preds = prediction.argmax(-1).tolist()

    all_preds.extend(preds)
    all_labels.extend(label.tolist())

    all_ids.extend(image_id.tolist())

acc = np.sum(np.array(all_preds) == np.array(all_labels))/len(all_labels)*100
print("Target Acc:{}".format(acc))

if not os.path.exists("pseudo_labels"):
    os.makedirs("pseudo_labels")

with open("pseudo_labels/{}_{}_{}.txt".format(name.lower(), source, target), "w") as fh:
    write_str = ""
    for pl, fid in tqdm(zip(all_preds, all_ids)):
        write_str += f"{fid} {pl}\n"
    fh.write(write_str)