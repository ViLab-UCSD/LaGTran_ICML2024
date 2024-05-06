from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler, get_cosine_schedule_with_warmup
import json
import torch
import torch.utils.data as data
import numpy as np
import random
from tqdm import tqdm
import os
import argparse

from loader.transforms import val_transform
from loader.ego4d_loader import Ego4dLoader
from loader.sampler import BalancedSampler
from metrics import accuracy, percls_accuracy

from torch.utils.tensorboard import SummaryWriter
     

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="GeoPlaces", choices=["ego4d_cooking"])
parser.add_argument("--source", type=str, default="asia")
parser.add_argument("--target", type=str, default="usa")
parser.add_argument("--n_iters", type=int, default=20000)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=5e-5)
parser.add_argument("--root_dir", type=str, default="data")
parser.add_argument("--sample", action="store_true")
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--backbone", type=str, default="distilbert-base-uncased")
parser.add_argument("--wd", type=float, default=3e-4)

## Sample command: python3 text_classification_distilbert.py --dataset GeoPlaces --source asia --target usa --n_iters 50000 --batch_size 32 --learning_rate 5e-5 --root_dir data

## Equivalent bash command: python3 text_classification_bert.py --dataset ${dataset} --source ${source} --target ${target} --n_iters ${n_iters} --batch_size ${batch_size} --learning_rate ${learning_rate} --root_dir ${root_dir}

args = parser.parse_args()

# cap_src = 'llm_cap_llama_13b'
cap_src = "text_caption"
# cap_src = 'blip2_cap'

if args.dataset == "GeoPlaces":
    N_CLASSES = 205
elif args.dataset == "GeoImnet":
    N_CLASSES = 600
elif args.dataset.lower() == "GeoYFCC".lower():
    N_CLASSES = 68
elif args.dataset.lower() == "GeoUniDA".lower():
    N_CLASSES = 62
elif args.dataset.lower() == "DomainNet".lower():
    N_CLASSES = 345
elif args.dataset.lower() == "ego4d_cooking".lower():
    N_CLASSES = 24

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def lower(s): return s.lower()

# def tag_preprocess(text, domain="sketch"):
#     #  return list(map(lambda t:t.replace(",", " "), text))
#     # return text
#     def rep(s):
#         s = s.replace(" a {} of".format(domain), "")
#         s = s.replace(" {} of".format(domain), "")
#         s = s.replace(" {}".format(domain), "")
#         return s
    
#     text = list(map(rep, text))
#     return text

def tag_preprocess(text):
    return text

BATCH_SIZE = args.batch_size
n_iters = args.n_iters
name = args.dataset
source = args.source
target = args.target
exp_name="traintest_{}_{}_{}_{}_sample_{}_{}_sgd".format(name, source, target, str(args.learning_rate).replace(".", "-"), args.sample, cap_src)

print("Saving to {}".format(exp_name))

tokenizer = AutoTokenizer.from_pretrained(args.backbone, model_max_length=256)
model = AutoModelForSequenceClassification.from_pretrained(args.backbone, num_labels=N_CLASSES).cuda()
# if args.backbone == "gpt2":
#     tokenizer.pad_token = tokenizer.eos_token
#     model.config.pad_token_id = model.config.eos_token_id

model.train()

dataset = Ego4dLoader(root_dir=args.root_dir, 
                        json_path=f"metadata/{name.lower()}.json",
                        domain=source,
                        split="train",
                        return_text=True,
                        _text_keys=[cap_src])


train_labels = dataset.target
## classwise separation of train-test
train_indices = []
for i in range(N_CLASSES):
    train_indices.extend(np.where(np.array(train_labels) == i)[0][:int(len(np.where(np.array(train_labels) == i)[0])*0.75)].tolist())
# train_indices = random.sample(range(len(dataset)), int(len(dataset)*0.75))
# train_dataset = data.Subset(dataset, train_indices)
train_dataset = dataset


test_indices = list(set(range(len(dataset))) - set(train_indices))
test_dataset = data.Subset(dataset, test_indices)

if args.sample:
    # import pdb; pdb.set_trace()
    sampler = BalancedSampler(train_dataset)
    shuffle = False
else:
    sampler = None
    shuffle = True

train = data.DataLoader(train_dataset, batch_size=BATCH_SIZE,  shuffle=shuffle, sampler=sampler,
                    drop_last=True, pin_memory=False, num_workers=4)
test = data.DataLoader(test_dataset, batch_size=BATCH_SIZE,  shuffle=False,
                    drop_last=False, num_workers=4)

target_dataset = Ego4dLoader(root_dir=args.root_dir,
                        json_path=f"metadata/{name.lower()}.json",
                        domain=target,
                        split="train",
                        return_text=True,
                        _text_keys=[cap_src])

# target_indices = random.sample(range(len(target_dataset)), int(len(target_dataset)*0.5))
# target_dataset = data.Subset(target_dataset, target_indices)

target_train = data.DataLoader(target_dataset, batch_size=BATCH_SIZE,  shuffle=False, drop_last=False, num_workers=4)

loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

print("Train dataset:{}, Test dataset:{}, Target dataset:{}".format(len(train_dataset), len(test_dataset), len(target_dataset)))

# params = list(model.pre_classifier.parameters()) + list(model.classifier.parameters())
params = model.parameters()
optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.wd)
# optimizer = torch.optim.SGD(params, lr=args.learning_rate, weight_decay=args.wd, momentum=0.9)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=n_iters
)

# lr_scheduler = get_cosine_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=0,
#     num_training_steps=n_iters
# )

writer = SummaryWriter(log_dir="tensorboard_logs/{}".format(exp_name))

if not os.path.exists(os.path.join("final_checkpoints", exp_name)):
    os.makedirs(os.path.join("final_checkpoints", exp_name))

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

    tokens = tokenizer(tag_preprocess(text[cap_src]), padding="longest", truncation=True, return_tensors="pt")
    tokens.update({"labels":label})
    tokens = {k:v.cuda() for k,v in tokens.items()}
    outputs = model(**tokens)
    # try:
    #     outputs = model(**tokens)
    # except:
    #     breakpoint()

    loss = outputs.loss
    writer.add_scalar("Loss/train", loss, it)

    loss.backward()
    optimizer.step()
    lr_scheduler.step()
         
    if (it + 1) % 400 == 0:
        
        model.eval()

        # all_preds = []
        # all_labels = []
        # for _, _, label, text in (iter(train)):

        #         tokens = tokenizer(tag_preprocess(text[cap_src]), padding="longest", truncation=True, return_tensors="pt")
        #         tokens = {k:v.cuda() for k,v in tokens.items()}
        #         with torch.no_grad():
        #             outputs = model(**tokens)

        #         prediction = outputs.logits.argmax(-1).tolist()
        #         gt_labels = label.tolist()

        #         all_preds.extend(prediction)
        #         all_labels.extend(gt_labels)

        # acc = np.sum(np.array(all_preds) == np.array(all_labels))/len(all_labels)*100
        # print("Iter:{}, Source Acc:{}".format(it+1, acc))
        # writer.add_scalar("Acc/train", acc, it)

        all_preds = []
        all_labels = []
        for _, _, label, text in (iter(test)):

                tokens = tokenizer(tag_preprocess(text[cap_src]), padding="longest",  truncation=True, return_tensors="pt")
                tokens = {k:v.cuda() for k,v in tokens.items()}
                with torch.no_grad():
                    outputs = model(**tokens)

                prediction = outputs.logits.argmax(-1).tolist()
                gt_labels = label.tolist()

                all_preds.extend(prediction)
                all_labels.extend(gt_labels)

        # acc = np.sum(np.array(all_preds) == np.array(all_labels))/len(all_labels)*100
        acc = percls_accuracy(all_preds, all_labels).mean()
        print("Iter:{}, Source Acc:{}".format(it+1, acc))
        writer.add_scalar("Acc/test", acc, it)

        all_preds = []
        all_labels = []
        for _, _, label, text in (iter(target_train)):
                
                tokens = tokenizer(tag_preprocess(text[cap_src]), padding="longest",  truncation=True, return_tensors="pt")
                tokens = {k:v.cuda() for k,v in tokens.items()}
                with torch.no_grad():
                    outputs = model(**tokens)

                prediction = outputs.logits#.argmax(-1).tolist()
                gt_labels = label.tolist()

                all_preds.append(prediction.detach())
                all_labels.extend(gt_labels)

        # acc = np.sum(np.array(all_preds) == np.array(all_labels))/len(all_labels)*100
        all_preds = torch.cat(all_preds, dim=0).cpu()
        top1, top5 = accuracy(all_preds, torch.tensor(all_labels), topk=(1,5))
        acc = percls_accuracy(all_preds.argmax(-1).tolist(), all_labels).mean()
        if top5 > best_acc:
            best_acc = top5
            print("Saving best model with acc:{}".format(best_acc))
            torch.save(model.state_dict(), "final_checkpoints/{}/best.pth".format(exp_name))
            best_model = model.state_dict()

        print("Iter:{}, Target Acc Top 1:{}, Top 5:{} \n".format(it+1, acc, top5), flush=True)
        writer.add_scalar("Acc/target", acc, it)

## compute target preditions using the best model.

print("\nComputing target predictions...")
model = AutoModelForSequenceClassification.from_pretrained(args.backbone, num_labels=N_CLASSES).cuda()
model.load_state_dict(torch.load("final_checkpoints/{}/best.pth".format(exp_name)))
model.eval()

all_preds = []
all_labels = []
all_flickrids = []

for fid, _, label, text in tqdm(target_train):
         
    tokens = tokenizer(tag_preprocess(text[cap_src]), padding="longest", truncation=True, return_tensors="pt")
    tokens = {k:v.cuda() for k,v in tokens.items()}
    with torch.no_grad():
        outputs = model(**tokens)

    prediction = outputs.logits.detach()#.argmax(-1).tolist()
    # soft_preds = torch.nn.functional.softmax(prediction/0.07, dim=-1)
    preds = prediction.argmax(-1).tolist()

    # all_preds.append(soft_preds.detach())
    all_preds.extend(preds)
    all_labels.extend(label.tolist())

    all_flickrids.extend(fid.tolist())

# all_preds = torch.cat(all_preds, dim=0).cpu()
## compute accuracy from all_preds and all_labels
acc = np.sum(np.array(all_preds) == np.array(all_labels))/len(all_labels)*100
print("Target Acc:{}".format(acc))
# print(accuracy(all_preds, torch.tensor(all_labels), topk=(1,5)))
write_str = ""
# soft_labels = all_preds.numpy()
# all_preds = all_preds.cpu().numpy()

with open("hard_labels/{}_{}_{}_blipPL.txt".format(name.lower(), source, target), "w") as fh:
    for pl, fid in tqdm(zip(all_preds, all_flickrids)):
        write_str += f"{fid} "
        write_str += f"{pl}"
        # vals = list(map(str, pl.tolist()))
        # write_str += " ".join(vals)
        write_str += "\n"
    fh.write(write_str)