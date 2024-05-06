from sentence_transformers import SentenceTransformer, util
import json
import torch
import torch.utils.data as data
import numpy as np
import random
from tqdm import tqdm
import os

from models.mlpcls import MLPCls
from models.linearcls import LinearCls
from loader.transforms import val_transform
from loader.json_loader import ImageJSONLoader

from torch.utils.tensorboard import SummaryWriter

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def lower(s): return s.lower()

N_CLASSES = 205
BATCH_SIZE=128
exp_name="example"
name = "GeoPlaces"

class sentence_classifier(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.backbone = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1').cuda()
        self.classifier = MLPCls(feat_size=[384, 512], n_class=N_CLASSES, bn=False).cuda()
        # self.classifier = LinearCls(768, n_class=N_CLASSES).cuda()
    
    def forward(self, batch):
        with torch.no_grad():
            embeddings = self.backbone.encode(batch, batch_size=BATCH_SIZE, convert_to_tensor=True)
            embeddings = embeddings.detach()
        outputs = self.classifier(embeddings)
        
        return outputs

model = sentence_classifier().cuda()
model.train()

dataset = ImageJSONLoader(root_dir=f"/home/tarun/metadata/{name}/", 
                        json_path=f"metadata/{name.lower()}.json",
                        transform=val_transform(256),
                        domain="usa",
                        split="train",
                        return_meta=True,
                        _meta_keys=["llm_cap_llama_13b"])

random_indices = random.sample(range(len(dataset)), 1000)
dataset = data.Subset(dataset, random_indices)

train = data.DataLoader(dataset, batch_size=BATCH_SIZE,  shuffle=True,
                    drop_last=True, pin_memory=False, num_workers=4)

target_dataset = ImageJSONLoader(root_dir=f"/home/tarun/metadata/{name}/", 
                        json_path=f"metadata/{name.lower()}.json",
                        transform=val_transform(256),
                        domain="asia",
                        split="train",
                        return_meta=True,
                        _meta_keys=["llm_cap_llama_13b"])

target_train = data.DataLoader(target_dataset, batch_size=BATCH_SIZE,  shuffle=False, drop_last=False, num_workers=4)

loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

optimizer = torch.optim.SGD([
    {'params':model.backbone.parameters(), "lr":0},
    {'params':model.classifier.parameters(), "lr":0.2}
], momentum=0.9)

writer = SummaryWriter(log_dir="tensorboard_logs/{}".format(exp_name))

iter_ds = iter(train)
n_iters = 50000
for it in range(n_iters):
    model.train()

    try:
        _, _, label, text = next(iter_ds)
    except:
        iter_ds = iter(train)
        _, _, label, text = next(iter_ds)
         
    optimizer.zero_grad()
    label = label.cuda()
    text = list(map(lower, text['llm_cap_llama_13b']))
    outputs = model(text)

    loss = loss_fn(outputs, label)
    writer.add_scalar("Loss/train", loss, it)

    loss.backward()
    optimizer.step()
         
    if (it+1) % 5000 == 0:
        
        model.eval()

        all_preds = []
        all_labels = []
        for _, _, label, text in (iter(train)):

                text = list(map(lower, text['llm_cap_llama_13b']))
                with torch.no_grad():
                    outputs = model(text)

                prediction = outputs.argmax(-1).tolist()
                gt_labels = label.tolist()

                all_preds.extend(prediction)
                all_labels.extend(gt_labels)

        acc = np.sum(np.array(all_preds) == np.array(all_labels))/len(all_labels)*100
        print("Iter:{}, Source Acc:{}".format(it+1, acc))

        all_preds = []
        all_labels = []
        for _, _, label, text in (iter(target_train)):

                text = list(map(lower, text['llm_cap_llama_13b']))
                with torch.no_grad():
                    outputs = model(text)

                prediction = outputs.argmax(-1).tolist()
                gt_labels = label.tolist()

                all_preds.extend(prediction)
                all_labels.extend(gt_labels)

        acc = np.sum(np.array(all_preds) == np.array(all_labels))/len(all_labels)*100
        print("Iter:{}, Target Acc:{}".format(it+1, acc))

