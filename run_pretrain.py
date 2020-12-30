# coding: utf-8
# from src.train_and_evaluate import *
from src.models import *
import os
import time
import math
import torch.optim
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup

from src.pre_data import *
from src.optim_schedule import *

USE_CUDA = True

max_seq_length = 340
batch_size = 12
test_batch_size = 1
embedding_size = 128
hidden_size = 768
n_epochs = 10
learning_rate_bert = 1e-4 #5e-5 #3e-5 # 2e-5
weight_decay_bert = 0.1 #1e-5 #2e-5 #2e-5 # 1e-5
betas=(0.9, 0.999)
warmup_proportion = 0.1 

save_path = "./model_ep%d_lrb%.0e_wdb%.0e" %(n_epochs, learning_rate_bert, weight_decay_bert)

bert_path = "/data3/yangzhicheng/Data/Pretrained_Model/bert-base-chinese"
# 创建save文件夹
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print("make dir ", save_path)

def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

def get_data_fold(data):
    data_ = []
    max_len = 0

    count = 0
    for line in data:
        num_pos = []

        # if count == 100:
        #     return data_, max_len
        count += 1

        for i in range(len(line["text_token"])):
            if line["text_token"][i] == "[NUM]":
                num_pos.append(i)
            max_len = max(max_len, len(''.join(line["text_token"])))
        data_.append([line["id"], line["text_token"], line["expression"], num_pos, line["group_num"], line["separate"]])
    return data_, max_len

# id, text_token, expression, num_pos, entity_pos, seperate
train_data, max_len_train = get_data_fold(read_json("data/train_processed.json"))
test_data, max_len_test = get_data_fold(read_json("data/test_processed.json"))

train_steps = n_epochs * math.ceil(len(train_data) / batch_size)

train_data = prepare_data_Bert(train_data, bert_path, max_seq_length)
test_data = prepare_data_Bert(test_data, bert_path, max_seq_length)


train_dataset = BERTDataset(train_data, bert_path, max_seq_length)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=3)

# data_iter = tqdm(enumerate(train_data_loader),
#                     desc="EP_%s:%d" % ("train", 1),
#                     total=len(train_data_loader),
#                     bar_format="{l_bar}{r_bar}")
# avg_loss = 0.0
# for i, data in data_iter:
#     data = {key: value for key, value in data.items()}
#     token_len = data["token_len"][0]
#     print(data["token_ids"][0][:token_len])
#     print(data["lm_labels"][0])
#     break


test_dataset = BERTDataset(test_data, bert_path, max_seq_length)
test_data_loader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=3)

# Initialize models
model = BERTLM(bert_path)


# param_optimizer = list(model.named_parameters())
# no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay_bert},
#         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#     ]
optimizer = AdamW(model.parameters(), #optimizer_grouped_parameters,
                  lr = learning_rate_bert, # args.learning_rate - default is 5e-5
                  betas=(.9, .999),
                  weight_decay=weight_decay_bert
                #   correct_bias = False
                )
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                    num_warmup_steps = int(train_steps * warmup_proportion), # Default value in run_glue.py
                                    num_training_steps = train_steps)

# optimizer = Lamb(model.parameters(), lr=learning_rate_bert, weight_decay=weight_decay_bert, betas=(.9, .999), adam=False)#(args.optimizer == 'adam'))

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_bert, weight_decay=weight_decay_bert, betas=betas)
# optim_schedule = ScheduledOptim(optimizer, model.config.hidden_size, n_warmup_steps=int(train_steps * warmup_proportion))

if USE_CUDA:
    model.cuda()

best_loss = 9999999
for epoch_ in range(n_epochs):
    epoch = epoch_ + 1

    ## ---------- Train ------------ ##
    data_iter = tqdm(enumerate(train_data_loader),
                     desc="EP_%s:%d" % ("train", epoch),
                     total=len(train_data_loader),
                     bar_format="{l_bar}{r_bar}")
    avg_loss = 0.0

    model.train()

    for i, data in data_iter:
        optimizer.zero_grad()
        if USE_CUDA:
            data = {key: value.cuda() for key, value in data.items()}
        else:
            data = {key: value for key, value in data.items()}
        loss = model(data["token_ids"], data["token_type_ids"], data["attention_mask"], data["lm_labels"])
        
        # optim_schedule.zero_grad()
        loss.backward()
        # optim_schedule.step_and_update_lr()
        optimizer.step()
        scheduler.step()

        avg_loss += loss.item()

        post_fix = {
            "epoch": epoch,
            "iter": i,
            "avg_loss": avg_loss / (i + 1),
            "loss": loss.item()
        }

        if i % 5 == 0:  # log_freq = 5
            data_iter.write(str(post_fix))

    print("EP%d_%s, avg_loss=" % (epoch, "train"), avg_loss / len(data_iter))

    ## Saving Model
    print("saving model...")
    model.savebert(save_path + "/pytorch_model_epoch" + str(epoch) + ".bin")

    ## ------------ Test ------------- ##
    data_iter = tqdm(enumerate(test_data_loader),
                     desc="EP_%s:%d" % ("train", epoch),
                     total=len(test_data_loader),
                     bar_format="{l_bar}{r_bar}")
    avg_loss = 0.0

    model.eval()

    for i, data in data_iter:
        if USE_CUDA:
            data = {key: value.cuda() for key, value in data.items()}
        else:
            data = {key: value for key, value in data.items()}
        loss = model(data["token_ids"], data["token_type_ids"], data["attention_mask"], data["lm_labels"])
        avg_loss += loss.item()

    print("EP%d_%s, avg_loss=" % (epoch, "Test"), avg_loss / len(data_iter))

    
    # if best_loss < avg_loss / len(data_iter):
    #     print("saving model...")
    #     best_loss = avg_loss / len(data_iter)
    #     model.savebert(save_path + "/pytorch_model.bin")
