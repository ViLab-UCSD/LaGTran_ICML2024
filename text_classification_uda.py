from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler
import json
import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
import random
from tqdm import tqdm
import os
import argparse

from loader.transforms import val_transform
from loader.json_loader import ImageJSONLoader
from loader.sampler import BalancedSampler
from metrics import accuracy

seed=1234
torch.manual_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

from torch.utils.tensorboard import SummaryWriter

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

def test(loader, model):

    model.eval()

    all_preds = []
    all_labels = []
    for _, _, label, text in (iter(loader)):
            
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

    return top1, top5
     

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="GeoPlaces", choices=["GeoPlaces", "GeoImnet", "GeoYfcc", "GeoUniDA", "DomainNet"])
parser.add_argument("--source", type=str, default="asia")
parser.add_argument("--target", type=str, default="usa")
parser.add_argument("--n_iters", type=int, default=20000)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=0.003)
parser.add_argument("--root_dir", type=str, default="data")
parser.add_argument("--sample", action="store_true")
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--backbone", type=str, default="distilbert-base-uncased")
parser.add_argument("--wd", type=float, default=3e-4)
parser.add_argument("--adapt", type=str, choices=["plain","dann","cdan"], default="plain")

## Sample command: python3 text_classification_distilbert.py --dataset GeoPlaces --source asia --target usa --n_iters 50000 --batch_size 32 --learning_rate 5e-5 --root_dir data

## Equivalent bash command: python3 text_classification_bert.py --dataset ${dataset} --source ${source} --target ${target} --n_iters ${n_iters} --batch_size ${batch_size} --learning_rate ${learning_rate} --root_dir ${root_dir}

args = parser.parse_args()

# cap_src = 'llm_cap_llama_13b'
# cap_src = "combined"
cap_src = 'blip2_cap'

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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BATCH_SIZE = args.batch_size
n_iters = args.n_iters
name = args.dataset
source = args.source
target = args.target
exp_name="{}_{}_{}_{}_sample_{}_{}_uda_{}".format(name, source, target, str(args.learning_rate).replace(".", "-"), args.sample, cap_src, args.adapt)

print("Saving to {}".format(exp_name))

#################
# Model
#################
tokenizer = AutoTokenizer.from_pretrained(args.backbone)
model = AutoModelForSequenceClassification.from_pretrained(args.backbone, num_labels=N_CLASSES).cuda()
if args.backbone == "gpt2":
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

model.train()
#################
#################

#################
# Discriminator
#################

class discriminator(nn.Module):
    def __init__(self, feature_len, total_classes=None):
        super().__init__()
        if total_classes is None:
            self.ad_layer1 = nn.Linear(feature_len, 1024)
        else:
            self.ad_layer1 = nn.Linear(feature_len * total_classes, 1024)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer1.bias.data.fill_(0.0)
        self.fc1 = nn.Sequential(self.ad_layer1, nn.ReLU(), nn.Dropout(0.5))

        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer3.bias.data.fill_(0.0)
        self.fc2_3 = nn.Sequential(self.ad_layer2, nn.ReLU(), nn.Dropout(0.5), self.ad_layer3)

    def forward(self, x, y=None):
        if y is not None:
            op_out = torch.bmm(y.unsqueeze(2), x.unsqueeze(1))
            ad_in = op_out.view(-1, y.size(1) * x.size(1))
        else:
            ad_in = x
        f2 = self.fc1(ad_in)
        f = self.fc2_3(f2)
        return f

class AdversarialLayer(torch.autograd.Function):
    iter_num = 0
    max_iter = 12500
    @staticmethod
    def forward(ctx, input):
        #self.iter_num += 1
#         ctx.save_for_backward(iter_num, max_iter)
        AdversarialLayer.iter_num += 1
        return input * 1.0

    @staticmethod
    def backward(ctx, gradOutput):
        alpha = 10
        low = 0.0
        high = 1.0
        lamb = 2.0
        iter_num, max_iter = AdversarialLayer.iter_num, AdversarialLayer.max_iter 
        # print('iter_num {}'.format(iter_num))
        coeff = float(lamb * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)
        return -coeff * gradOutput
    
if args.adapt == "dann":
    discriminator = discriminator(768).cuda()
    discriminator.train()
elif args.adapt == "cdan":
    discriminator = discriminator(768, N_CLASSES).cuda()
    discriminator.train()
else:
    discriminator = None

grl = AdversarialLayer()
###################
###################

criterion = nn.BCEWithLogitsLoss()

#################
# Dataset
#################
dataset = ImageJSONLoader(root_dir=args.root_dir, 
                        json_path=f"metadata/{name.lower()}.json",
                        transform=val_transform(256),
                        domain=source,
                        split="train",
                        return_meta=True,
                        _meta_keys=[cap_src])

train_indices = random.sample(range(len(dataset)), int(len(dataset)*0.7))
train_dataset = dataset #data.Subset(dataset, train_indices)

test_indices = list(set(range(len(dataset))) - set(train_indices))
test_dataset = data.Subset(dataset, test_indices)

if args.sample:
    # import pdb; pdb.set_trace()
    sampler = BalancedSampler(train_dataset)
    shuffle = False
else:
    sampler = None
    shuffle = True

train_ds = data.DataLoader(train_dataset, batch_size=BATCH_SIZE,  shuffle=shuffle, sampler=sampler,
                    drop_last=True, pin_memory=False, num_workers=4)
test_ds = data.DataLoader(test_dataset, batch_size=BATCH_SIZE,  shuffle=False,
                    drop_last=False, num_workers=4)

target_dataset = ImageJSONLoader(root_dir=args.root_dir,
                        json_path=f"metadata/{name.lower()}.json",
                        transform=val_transform(256),
                        domain=target,
                        split="train",
                        return_meta=True,
                        _meta_keys=[cap_src])

target_test = ImageJSONLoader(root_dir=args.root_dir,
                        json_path=f"metadata/{name.lower()}.json",
                        transform=val_transform(256),
                        domain=target,
                        split="test",
                        return_meta=True,
                        _meta_keys=[cap_src])

# target_indices = random.sample(range(len(target_dataset)), int(len(target_dataset)*0.5))
# target_dataset = data.Subset(target_dataset, target_indices)
target_ds = data.DataLoader(target_dataset, batch_size=BATCH_SIZE,  shuffle=False, drop_last=False, num_workers=4)
target_test_ds = data.DataLoader(target_test, batch_size=BATCH_SIZE,  shuffle=False, drop_last=False, num_workers=4)

print("Train dataset:{}, Test dataset:{}, Target dataset:{}, Target Test:{}".format(len(train_dataset), len(test_dataset), len(target_dataset), len(target_test)))
#################
#################

loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")


# params = list(model.pre_classifier.parameters()) + list(model.classifier.parameters())
# params = model.parameters()
params = [
    {'params': model.parameters(), "lr":args.learning_rate},
]

if discriminator is not None:
    params += [
        {'params': discriminator.parameters(), "lr":args.learning_rate},    
    ]

# optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.wd)
optimizer = torch.optim.SGD(params, weight_decay=args.wd, momentum=0.9)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=n_iters
)

writer = SummaryWriter(log_dir="tensorboard_logs/{}".format(exp_name))

if not os.path.exists(os.path.join("final_checkpoints", exp_name)):
    os.makedirs(os.path.join("final_checkpoints", exp_name))

best_acc = 0
source_ds = iter(train_ds)
target = iter(target_ds)
for it in range(n_iters):
    model.train()
    optimizer.zero_grad()

    try:
        _, _, label, text = next(source_ds)
    except:
        source_ds = iter(train_ds)
        _, _, label, text = next(source_ds)

    try:
        _, _, _, text_target = next(target)
    except:
        target = iter(target_ds)
        _, _, _, text_target = next(target)
         

    tokens = tokenizer(tag_preprocess(text[cap_src]), padding="longest", truncation=True, return_tensors="pt")
    tokens.update({"labels":label})
    tokens = {k:v.cuda() for k,v in tokens.items()}
    tokens.update({
        'output_hidden_states': True,
        'return_dict': True
    })
    outputs_source = model(**tokens)
    logits_source = outputs_source.logits
    hidden_features_source = outputs_source.hidden_states[-1][:,0] ## pass this through adversarial network.
    classifier_loss = outputs_source.loss

    tokens = tokenizer(tag_preprocess(text_target[cap_src]), padding="longest", truncation=True, return_tensors="pt")
    tokens = {k:v.cuda() for k,v in tokens.items()}
    tokens.update({
        'output_hidden_states': True,
        'return_dict': True
    })
    outputs_target = model(**tokens)
    logits_target = outputs_target.logits
    hidden_features_target = outputs_target.hidden_states[-1][:,0] ## pass this through adversarial network.
    
    domain_labels = torch.tensor([[1], ] * len(hidden_features_source)+ [[0], ] * len(hidden_features_target), device=torch.device('cuda:0'), dtype=torch.float)
    logits = torch.cat([logits_source, logits_target], dim=0).detach()
    features = torch.cat([hidden_features_source, hidden_features_target], dim=0)

    if args.adapt != "plain":
        if args.adapt == "dann":
            domain_predicted = discriminator(grl.apply(features))
        elif args.adapt == "cdan": 
            domain_predicted = discriminator(grl.apply(features), torch.softmax(logits, dim=-1))
        transfer_loss = criterion(domain_predicted, domain_labels)
        loss = classifier_loss + 0.1*transfer_loss*(it > 1000)
        writer.add_scalar("Loss/transfer", transfer_loss.item(), it)
    else:
        loss = classifier_loss
    writer.add_scalar("Loss/train", classifier_loss.item(), it)

    loss.backward()
    optimizer.step()
    lr_scheduler.step()
         
    if (it + 1) % 2000 == 0:

        top1, top5 = test(test_ds, model)

        print("Iter:{}, Source Acc:{}".format(it+1, top1))
        writer.add_scalar("Acc/test", top1, it)

        top1, top5 = test(target_test_ds, model)
        if top5 > best_acc:
            best_acc = top5
            print("Saving best model with acc:{}".format(best_acc))
            torch.save(model.state_dict(), "final_checkpoints/{}/best.pth".format(exp_name))
            best_model = model.state_dict()

        print("Iter:{}, Target Acc Top 1:{}, Top 5:{} \n".format(it+1, top1, top5), flush=True)
        writer.add_scalar("Acc/target", top1, it)

## compute target preditions using the best model.
# import sys
# sys.exit(0)

print("\nComputing target predictions...")
model = AutoModelForSequenceClassification.from_pretrained(args.backbone, num_labels=N_CLASSES).cuda()
model.load_state_dict(torch.load("final_checkpoints/{}/best.pth".format(exp_name)))
model.eval()

all_preds = []
all_labels = []
all_flickrids = []

for fid, _, label, text in tqdm(target_ds):
         
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

with open("soft_labels/{}_{}_{}_{}PL.txt".format(name.lower(), source, target, args.adapt), "w") as fh:
    for pl, fid in tqdm(zip(all_preds, all_flickrids)):
        write_str += f"{fid} "
        write_str += f"{pl}"
        # vals = list(map(str, pl.tolist()))
        # write_str += " ".join(vals)
        write_str += "\n"
    fh.write(write_str)