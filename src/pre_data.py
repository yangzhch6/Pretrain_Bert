import random
import json
import copy
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertConfig, BertModel

PAD_token = 0

class BERTDataset(Dataset):
    # data: token_ids, token_type_ids, attention_mask, token_len, word_idx, num_pos, num_token_idx, group, separate 
    def __init__(self, data, bert_path, max_seq_length):
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        line = self.data[item]
        token_ids, token_type_ids, attention_mask, token_len, word_idx, num_pos, num_token_idx, group, separate = line
        token_ids, lm_labels = self.random_word(token_ids, token_len, group)


        output = {"token_ids": token_ids,
                  "token_type_ids": token_type_ids,
                  "attention_mask": attention_mask,
                  "lm_labels": lm_labels,
                  "token_len": token_len}

        return {key: torch.tensor(value) for key, value in output.items()}


    def random_word(self, token_ids_, token_len, group):
        token_ids = copy.deepcopy(token_ids_)
        output_label = [-100]*self.max_seq_length

        for i in range(1, token_len-1):
            token_id = token_ids[i]

            if token_id != 1:
                threshold = 0.15
            else:
                threshold = 0.15
                
            prob = random.random()
            if prob < threshold:
                prob /= threshold
                # 80% randomly change token to mask token
                if prob < 0.8:
                    token_ids[i] = self.tokenizer.mask_token_id
                # 10% randomly change token to random token
                elif prob < 0.9:
                    token_ids[i] = random.randrange(self.tokenizer.vocab_size)
                # 10% randomly change token to current token
                # else:
                #     pass
                output_label[i] = token_id
            # else:
            #     pass
        return token_ids, output_label


# Return a list of indexes, one for each word in the sentence, plus EOS
def cut_input(sentence, num_pos_, group_, separate_):
    res = []
    num_pos = num_pos_
    group = group_
    separate = separate_
    idx = 0
    for word in sentence:
        if len(word) == 0:
            # -------------------------------------
            i = 0  # i负责遍历
            while i < len(num_pos):
                if num_pos[i] == idx:
                    num_pos.pop(i)
                elif num_pos[i] > idx:
                    num_pos[i] -= 1
                    i += 1
                else:
                    i += 1
            # -------------------------------------
            i = 0
            while i < len(group):
                if group[i] == idx:
                    group.pop(i)
                elif group[i] > idx:
                    group[i] -= 1
                    i += 1
                else:
                    i += 1
            # -------------------------------------
            i = 0
            while i < len(separate):
                if separate[i] == idx:
                    separate.pop(i)
                elif separate[i] > idx:
                    separate[i] -= 1
                    i += 1
                else:
                    i += 1
            # -------------------------------------
            continue
        idx += 1
    
    return num_pos, group, separate

# data: [id, text_token, expression, num_pos, entity_pos, seperate]
def prepare_data_Bert(ori_data, path, max_seq_length):
    train_pairs = []
    # Load Bert-chinese-wwm tokenizer 
    tokenizer = BertTokenizer.from_pretrained(path)
    PAD_id = tokenizer.pad_token_id

    for data in ori_data:
        num_pos, group, separate = cut_input(data[1], data[3], data[4], data[5])

        tokens = [] 
        word_idx = [] 
        start = 1
        for d in data[1]:
            if d == '':
                continue
            elif d == '[NUM]':
                word_idx.append([start])
                start += 1
                tokens.append(d)
            elif d != '[NUM]':
                tmp_token= tokenizer.tokenize(d) #[UNK]
                if tmp_token == [] or len(tmp_token) == 0:
                    print("processing ", d, [d])
                    tmp_token = ['[UNK]']
                tokens += tmp_token
                tmp = np.linspace(start, start+len(tmp_token)-1, num=len(tmp_token)).astype(np.int16).tolist()
                if tmp == []:
                    print(d, [d])
                    print(tmp_token)
                    print(tokens)
                    print(data[1], len(data[1]))
                    print("_______________________________________")
                start += len(tmp_token)
                word_idx.append(tmp)
                    
        tokens = ['[CLS]'] + tokens + ['[SEP]'] 
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_len = len(token_ids)
        # Padding 
        padding_ids = [PAD_id]*(max_seq_length - len(token_ids))
        token_ids += padding_ids
        # token_type_ids
        token_type_ids = [0]*max_seq_length
        # attention_mask
        attention_mask = [1]*token_len + padding_ids

        num_token_idx = []
        for idx in num_pos:
            num_token_idx += word_idx[idx]

        print_ = False
        if token_len != word_idx[-1][-1]+2:
            print(tokens)
            print(data[1])
            print(word_idx)
            print(token_len, word_idx[-1][-1]+2)
            print_ = True
        if print_:
            print("————————————————————————————————————————————————————————————————————————————————")
        
    
        if len(num_token_idx) != len(num_pos):
            print(data[1])
            print(tokens)
            print(num_token_idx)
            # print(data[2])
            print(num_pos)
            print(word_idx)
        
        ### Testing num token
        for id in num_pos:
            num_id = word_idx[id][0]
            if len(word_idx[id]) != 1 or tokens[num_id] != '[NUM]':
                print(data[1])
                print(num_id, data[1][num_id])
                print(num_pos)
                print(tokens)
                print(word_idx)
                print("————————————————————————————————————————————————————————————————————————————————")
                break

        train_pairs.append((token_ids, token_type_ids, attention_mask, token_len, word_idx, num_pos, num_token_idx, group, separate))

    print('Number of training data %d' % (len(train_pairs)))

    return train_pairs

