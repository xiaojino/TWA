import torch
import  editdistance
import collections
import random
import copy
import numpy as np
from pytorch_transformers.tokenization_bert import BertTokenizer
chars = [' ', '<s>', '</s>', '<unk>', '{', "'", 'g', '<', 'é', ';', '5', 'p', '-', ')', '7', 'v', '(', 'x', '$', '/', 'u', 'f', '^', 'q', '₹', 'i', ']', '!', '#', '❞', 'a', '£', '3', 'n', 'w', '0', '?', '8', '+', 's', 'œ', '©', '1', '½', '.', 'l', '&', '↓', '>', '9', 'z', 'c', '[', '_', 'ñ', 'e', ':', 'b', 'è', 'm', '●', '²', '"', '~', 'ð', '°', '✓', 'r', 'o', '2', '®', 'h', '@', 'σ', '•', 't', 'd', '6', '}', '€', 'y', '*', '=', '¿', '¥', '%', '÷', 'j', '|', '4', '�', '→', '←', 'k', ',', '™']
char_set = {c:idx for idx,c in enumerate(chars)}
bert_tokenizer = BertTokenizer.from_pretrained('data/bert-base-uncased')
assert bert_tokenizer.encode(bert_tokenizer.pad_token) == [0]


def load_line_to_ids_dict(fname):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(fname, "r", encoding="utf-8") as reader:
        chars = reader.readlines()
    for index, char in enumerate(chars):
        char = char.rstrip('\n')
        vocab[char] = index
    return vocab
def load_ids_to_line_dict(fname):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(fname, "r", encoding="utf-8") as reader:
        chars = reader.readlines()
    for index, char in enumerate(chars):
        char = char.rstrip('\n')
        vocab[index] = char
    return vocab
char2ids_dict = load_line_to_ids_dict(fname='data/dict/bert_char_vocab')
term2ids_dict = load_line_to_ids_dict(fname='data/dict/term_vocab')

def fun_recover_words():
    id2terms_dict = load_ids_to_line_dict(fname='data/dict/term_vocab')
    return id2terms_dict
#term2ids_dict = load_line_to_ids_dict(fname='data/bert-base-uncased/vocab.txt')
def add_ocr_char_info(ocr_tokens, ans_list, ocr_max_num, char_max_num=50, use_char_info=True, answer=False):
    assert answer == False
    if not use_char_info:
        if answer:
            return None, None, None
        else:
            return None, None
    if answer:
        assert len(ocr_tokens) == 1
        prev_indx = torch.zeros(ocr_max_num, char_max_num, dtype=torch.long)
    answer_mask = torch.zeros(ocr_max_num, char_max_num)
    if answer:
        ocr_char = 0 - torch.ones(ocr_max_num, char_max_num, dtype=torch.long)
    else:
        ocr_char = torch.zeros(ocr_max_num, char_max_num, dtype=torch.long)
    char_lable = torch.ones(ocr_max_num, dtype=torch.long)*(-1)
    _ocr_num = min(len(ocr_tokens), ocr_max_num)
    for idx in range(_ocr_num):
        min_score=1000
        max_ans=""
        token = ocr_tokens[idx].lower().strip()
        for ans in ans_list:
            ans = ans.lower().strip()
            nl = editdistance.eval(token, ans)
            if nl < min_score:
                min_score = nl
                max_ans = ans
        if min_score <= 2 and min_score!=0:
            if min_score<len(max_ans) and len(token)>1:
                if max_ans.lower() in term2ids_dict:
                    char_lable[idx] = term2ids_dict[max_ans.lower()]
                else:
                    char_lable[idx] = term2ids_dict['<unk>']
            
        if ocr_tokens[idx] == '<pad>':
            if len(ocr_tokens) > 500:
                continue
            else:
                break
        if ocr_tokens[idx] in ['<s>', '</s>', '<unk>']:
            continue
        _char_num = min(len(ocr_tokens[idx]), char_max_num)
        c_i = -1
        for c_i in range(_char_num):
            c = ocr_tokens[idx][c_i]
            if c in char_set:
                c_idx = char_set[c]
            else:
                c_idx = char_set['<unk>']
            ocr_char[idx][c_i] = c_idx
            answer_mask[idx][c_i] = 1
            if answer:
                if c_i == 0:
                    prev_indx[idx][c_i] = 1
                else:
                    prev_indx[idx][c_i] = ocr_char[idx][c_i-1]
        if answer:
            c_i += 1
            c_idx = char_set['</s>']
            if c_i < char_max_num:
                ocr_char[idx][c_i] = c_idx
                answer_mask[idx][c_i] = 1
                if c_i == 0:
                    prev_indx[idx][c_i] = 1
                else:
                    prev_indx[idx][c_i] = ocr_char[idx][c_i-1]
    if answer:
        return ocr_char, answer_mask, prev_indx
    return ocr_char, answer_mask, char_lable

def create_adv_word_adr(orig_token, rng):
    token = list(copy.deepcopy(orig_token))
    if len(orig_token) < 4:
        rand_idx = rng.randint(0, 80)
        rand_char = list(char2ids_dict.keys())[rand_idx]
        insert_idx = rng.randint(0, len(orig_token) - 1)
        token = token[:insert_idx] + [rand_char] + token[insert_idx:]
        # if self.args.output_debug and False:
        #    print(f"Insert the char:{rand_char} orig_token: {orig_token} new_token: {token}")
    else:
        if rng.random() < 0.5:
            rand_idx = rng.randint(0, len(orig_token) - 1)
            del token[rand_idx]
            # if self.args.output_debug and False:
            #    print(f"Delete the char:{orig_token[rand_idx]} orig_token: {orig_token} new_token: {token}")
        else:
            rand_idx = rng.randint(0, 80)
            rand_char = list(char2ids_dict.keys())[rand_idx]
            replace_idx = rng.randint(0, len(orig_token)-1)
            token[replace_idx] = rand_char
    token = ''.join(token)
    return token

def create_adv_word(orig_token, rng):
    token = list(copy.deepcopy(orig_token))
    if len(orig_token) < 4:
        rand_idx = rng.randint(0, 80)
        rand_char = list(char2ids_dict.keys())[rand_idx]
        insert_idx = rng.randint(0, len(orig_token) - 1)
        token = token[:insert_idx] + [rand_char] + token[insert_idx:]
        # if self.args.output_debug and False:
        #    print(f"Insert the char:{rand_char} orig_token: {orig_token} new_token: {token}")
    else:
        if rng.random() < 0.5:
            rand_idx = rng.randint(0, len(orig_token) - 1)
            del token[rand_idx]
            # if self.args.output_debug and False:
            #    print(f"Delete the char:{orig_token[rand_idx]} orig_token: {orig_token} new_token: {token}")
        else:
            idx = random.randint(1, len(orig_token) - 2)
            token[idx], token[idx + 1] = token[idx + 1], token[idx]
            # if self.args.output_debug and False:
            #    print(f"Swap the char:{token[idx:idx+2]} orig_token: {orig_token} new_token: {token}")
    token = ''.join(token)
    return token

def create_adv_word_len(orig_token, rng):
    token = list(copy.deepcopy(orig_token))
    if len(orig_token) < 4:
        rand_idx = rng.randint(0, 80)
        rand_char = list(char2ids_dict.keys())[rand_idx]
        insert_idx = rng.randint(0, len(orig_token) - 1)
        token = token[:insert_idx] + [rand_char] + token[insert_idx:]
        edit_len = 1
        # if self.args.output_debug and False:
        #    print(f"Insert the char:{rand_char} orig_token: {orig_token} new_token: {token}")
    else:
        if rng.random() < 0.4:
            rand_idx = rng.randint(0, len(orig_token) - 1)
            del token[rand_idx]
            edit_len=1
        elif  rng.random() < 0.7:
            idx = random.randint(1, len(orig_token) - 2)
            token[idx], token[idx + 1] = token[idx + 1], token[idx]
            edit_len = 2
        else:
            rand_idx = rng.randint(0, 80)
            rand_char = list(char2ids_dict.keys())[rand_idx]
            replace_idx = rng.randint(0, len(orig_token) - 1)
            token[replace_idx] = rand_char.lower()
            edit_len = 1
    token = ''.join(token)
    return token, edit_len

def find_related_word(token, term2ids_dict, editlen):
    min_score=10
    max_ans=""
    token = token.lower().strip()
    for ans in term2ids_dict:
        ans = ans.lower().strip()
        nl = editdistance.eval(token, ans)
        if nl < min_score:
            min_score = nl
            max_ans = ans
        if nl==0:
            min_score = nl
            max_ans = ans
            break
    if min_score <= editlen and min_score < len(max_ans) and len(token) > 1:
        word=max_ans
    else:
        word = token
    return word

def find_related_word_len(token, term2ids_dict):
    min_score=10
    max_ans=""
    token = token.lower().strip()
    for ans in term2ids_dict:
        ans = ans.lower().strip()
        nl = editdistance.eval(token, ans)
        if nl < min_score:
            min_score = nl
            max_ans = ans
        if nl==0:
            min_score = nl
            max_ans = ans
            break
    if min_score <= 2 and min_score < len(max_ans) and len(token) > 1:
        word=max_ans
        edit_len = min_score
    else:
        word = token
        edit_len = 0
    return word, edit_len

def findRelatedOCR_only(ocr_tokens, ocr_max_num,adv_probability):
    rng = random.Random(13)
    ocr_tokens = ocr_tokens[:ocr_max_num]
    _ocr_num = min(len(ocr_tokens), ocr_max_num)
    related_ocr_tokens = []
    for idx in range(_ocr_num):
        token = ocr_tokens[idx].lower().strip()
        randN = rng.random()
        if randN <= adv_probability:
            rel_token = find_related_word(token, term2ids_dict)
        else:
            rel_token = token
        related_ocr_tokens.append(rel_token)
    while len(ocr_tokens)<ocr_max_num:
        ocr_tokens.append(bert_tokenizer.pad_token)
        related_ocr_tokens.append(bert_tokenizer.pad_token)
    assert len(ocr_tokens)==len(related_ocr_tokens)
    return ocr_tokens, related_ocr_tokens
def get_anls(nl, s1, s2):
    iou = 1 - nl/ max(len(s1), len(s2))
    anls = iou if iou >= .5 else 0.
    return anls
def findRelatedOCR_ori(ocr_tokens, ocr_max_num,adv_probability,label_list,editlen):
    o2r_labels = torch.ones(ocr_max_num, ocr_max_num, dtype=torch.float) * (-1.0)
    r2o_labels = torch.ones(ocr_max_num, ocr_max_num, dtype=torch.float) * (-1.0)
    rng = random.Random(13)
    ocr_tokens = ocr_tokens[:ocr_max_num]
    _ocr_num = min(len(ocr_tokens), ocr_max_num)
    related_ocr_tokens = []
    for idx in range(_ocr_num):
        token = ocr_tokens[idx].lower().strip()
        randN = rng.random()
        if randN < adv_probability and len(token)>1:
            if token in term2ids_dict:
                #rel_token = create_adv_word_adr(token, rng)
                rel_token = create_adv_word(token, rng)
                o2r_labels[idx][idx] = label_list[1] #0.5
                r2o_labels[idx][idx] = label_list[0] #0.8
            else:
                rel_token = find_related_word(token, term2ids_dict,editlen)
                o2r_labels[idx][idx] = label_list[0]
                r2o_labels[idx][idx] = label_list[1]
                if rel_token.lower() == token.lower():
                    o2r_labels[idx][idx] = 1.0
                    r2o_labels[idx][idx] = 1.0
        else:
            rel_token = token
            o2r_labels[idx][idx] = 1.0
            r2o_labels[idx][idx] = 1.0
        related_ocr_tokens.append(rel_token)
    for i in range(_ocr_num):
        for j in range(i + 1, _ocr_num):
            if ocr_tokens[i].lower() == ocr_tokens[j].lower():
                o2r_labels[i][j] = o2r_labels[j][j]
                o2r_labels[j][i] = o2r_labels[i][i]
                r2o_labels[i][j] = r2o_labels[j][j]
                r2o_labels[j][i] = r2o_labels[i][i]
            else:
                o2r_labels[i][j] = 0.0
                o2r_labels[j][i] = 0.0
                r2o_labels[i][j] = 0.0
                r2o_labels[j][i] = 0.0
    while len(ocr_tokens)<ocr_max_num:
        ocr_tokens.append(bert_tokenizer.pad_token)
        related_ocr_tokens.append(bert_tokenizer.pad_token)
    assert len(ocr_tokens)==len(related_ocr_tokens)
    return ocr_tokens, related_ocr_tokens, o2r_labels, r2o_labels
def findRelatedOCR_adr(ocr_tokens, ocr_max_num,adv_probability,label_list,editlen):
    o2r_labels = torch.ones(ocr_max_num, ocr_max_num, dtype=torch.float) * (-1.0)
    r2o_labels = torch.ones(ocr_max_num, ocr_max_num, dtype=torch.float) * (-1.0)
    rng = random.Random(13)
    ocr_tokens = ocr_tokens[:ocr_max_num]
    _ocr_num = min(len(ocr_tokens), ocr_max_num)
    related_ocr_tokens = []
    for idx in range(_ocr_num):
        token = ocr_tokens[idx].lower().strip()
        randN = rng.random()
        if randN < adv_probability and len(token)>1:
            if token in term2ids_dict:
                rel_token = create_adv_word_adr(token, rng)
                o2r_labels[idx][idx] = label_list[1] #0.5
                r2o_labels[idx][idx] = label_list[0] #0.8
            else:
                rel_token = find_related_word(token, term2ids_dict,editlen)
                o2r_labels[idx][idx] = label_list[0]
                r2o_labels[idx][idx] = label_list[1]
                if rel_token.lower() == token.lower():
                    o2r_labels[idx][idx] = 1.0
                    r2o_labels[idx][idx] = 1.0
        else:
            rel_token = token
            o2r_labels[idx][idx] = 1.0
            r2o_labels[idx][idx] = 1.0
        related_ocr_tokens.append(rel_token)
    for i in range(_ocr_num):
        for j in range(i + 1, _ocr_num):
            if ocr_tokens[i].lower() == ocr_tokens[j].lower():
                o2r_labels[i][j] = o2r_labels[j][j]
                o2r_labels[j][i] = o2r_labels[i][i]
                r2o_labels[i][j] = r2o_labels[j][j]
                r2o_labels[j][i] = r2o_labels[i][i]
            else:
                o2r_labels[i][j] = 0.0
                o2r_labels[j][i] = 0.0
                r2o_labels[i][j] = 0.0
                r2o_labels[j][i] = 0.0
    while len(ocr_tokens)<ocr_max_num:
        ocr_tokens.append(bert_tokenizer.pad_token)
        related_ocr_tokens.append(bert_tokenizer.pad_token)
    assert len(ocr_tokens)==len(related_ocr_tokens)
    return ocr_tokens, related_ocr_tokens, o2r_labels, r2o_labels

def findRelatedOCR(ocr_tokens, ocr_max_num,adv_probability, editlen):
    rng = random.Random(13)
    ocr_tokens = ocr_tokens[:ocr_max_num]
    _ocr_num = min(len(ocr_tokens), ocr_max_num)
    related_ocr_tokens = []
    for idx in range(_ocr_num):
        token = ocr_tokens[idx].lower().strip()
        randN = rng.random()
        if randN < adv_probability and len(token)>1:
            if token in term2ids_dict:
                rel_token = create_adv_word_adr(token, rng)
                #rel_token = create_adv_word(token, rng)
            else:
                rel_token = find_related_word(token, term2ids_dict, editlen)
        else:
            rel_token = token
        related_ocr_tokens.append(rel_token)
    while len(ocr_tokens)<ocr_max_num:
        ocr_tokens.append(bert_tokenizer.pad_token)
        related_ocr_tokens.append(bert_tokenizer.pad_token)
    assert len(ocr_tokens)==len(related_ocr_tokens)
    return ocr_tokens, related_ocr_tokens


def object_bert_ids(obj_tokens,obj_max_num):
    word_ids = torch.zeros(obj_max_num, dtype=torch.long)
    for idx in range(obj_max_num):
        assert bert_tokenizer.encode(bert_tokenizer.pad_token) == [0]
        token = obj_tokens[idx].lower().strip()
        tokenized = bert_tokenizer.encode(token)
        if len(tokenized) > 0:
            word_ids[idx] = torch.LongTensor(tokenized[:1])
    return word_ids

def add_cons_ocr_info(ocr_tokens, ocr_max_num, char_max_num=50):
    rng = random.Random(13)
    ocr_char_mask = torch.zeros(ocr_max_num, char_max_num)
    ocr_char = torch.zeros(ocr_max_num, char_max_num, dtype=torch.long)
    _ocr_num = min(len(ocr_tokens), ocr_max_num)
    word_ids = torch.zeros(ocr_max_num, dtype=torch.long)
    for idx in range(_ocr_num):
        assert bert_tokenizer.encode(bert_tokenizer.pad_token) == [0]
        if ocr_tokens[idx] == bert_tokenizer.pad_token:
            continue
        token = ocr_tokens[idx].lower().strip()
        #if token == '<pad>':
        #    if len(ocr_tokens) > 500:
        #        continue
        #    else:
        #        break
        if token in ['<s>', '</s>', '<unk>','<pad>', '[pad]']:
            continue
        tokenized = bert_tokenizer.encode(token)
        if len(tokenized) > 0:
            word_ids[idx] = torch.LongTensor(tokenized[:1])

        _char_num = min(len(token), char_max_num)
        c_i = -1
        for c_i in range(_char_num):
            c = token[c_i]  # ocr_tokens[idx][c_i]
            if c in char_set:
                c_idx = char_set[c]
            else:
                c_idx = char_set['<unk>']
            ocr_char[idx][c_i] = c_idx
            ocr_char_mask[idx][c_i] = 1
    return ocr_char, ocr_char_mask, word_ids

def create_ocr_char_info(ocr_tokens, ocr_max_num, adv_probability, char_max_num=50, use_char_info=True, answer=False): 
    rng = random.Random(13)
    #adv_probability = 0.5 #0.15
    assert answer == False
    if not use_char_info:
        if answer:
            return None, None, None
        else:
            return None, None
    if answer:
        assert len(ocr_tokens) == 1
        prev_indx = torch.zeros(ocr_max_num, char_max_num, dtype=torch.long)
    answer_mask = torch.zeros(ocr_max_num, char_max_num)
    if answer:
        ocr_char = 0 - torch.ones(ocr_max_num, char_max_num, dtype=torch.long)
    else:
        ocr_char = torch.zeros(ocr_max_num, char_max_num, dtype=torch.long)
    char_lable = torch.ones(ocr_max_num, dtype=torch.long)*(-1)
    _ocr_num = min(len(ocr_tokens), ocr_max_num)
    word_ids = torch.zeros(ocr_max_num, dtype=torch.long)
    for idx in range(_ocr_num):
        token = ocr_tokens[idx].lower().strip()
        if token == '<pad>':
            if len(ocr_tokens) > 500:
                continue
            else:
                break
        if token in ['<s>', '</s>', '<unk>']:
            continue
       
        ori_token = copy.deepcopy(token)
        randN = rng.random()
        if randN < adv_probability:  # self.adv_probability: #self.args.adv_probability:
            if ori_token.lower() in term2ids_dict:
                token = create_adv_word(token, rng)
            else:
                ori_token = find_related_word(token,term2ids_dict)
        if ori_token != token:
            if ori_token.lower() in term2ids_dict:
                char_lable[idx] = term2ids_dict[ori_token.lower()]
            else:
                char_lable[idx] = term2ids_dict['[UNK]']  #term2ids_dict['<unk>'] 
        tokenized = bert_tokenizer.encode(ori_token)
        if len(tokenized)>0:
            word_ids[idx] = torch.LongTensor(tokenized[:1])
            
        
        _char_num = min(len(token), char_max_num)
        c_i = -1
        for c_i in range(_char_num):
            c = token[c_i] #ocr_tokens[idx][c_i]
            if c in char_set:
                c_idx = char_set[c]
            else:
                c_idx = char_set['<unk>']
            ocr_char[idx][c_i] = c_idx
            answer_mask[idx][c_i] = 1
            if answer:
                if c_i == 0:
                    prev_indx[idx][c_i] = 1
                else:
                    prev_indx[idx][c_i] = ocr_char[idx][c_i-1]
        if answer:
            c_i += 1
            c_idx = char_set['</s>']
            if c_i < char_max_num:
                ocr_char[idx][c_i] = c_idx
                answer_mask[idx][c_i] = 1
                if c_i == 0:
                    prev_indx[idx][c_i] = 1
                else:
                    prev_indx[idx][c_i] = ocr_char[idx][c_i-1]
    if answer:
        return ocr_char, answer_mask, prev_indx
    return ocr_char, answer_mask, char_lable, word_ids

def create_ocr_char_info2(ocr_tokens, ocr_max_num, adv_probability, char_max_num=50, use_char_info=True, answer=False): 
    rng = random.Random(13)
    #adv_probability = 0.5 #0.15
    assert answer == False
    if not use_char_info:
        if answer:
            return None, None, None
        else:
            return None, None
    if answer:
        assert len(ocr_tokens) == 1
        prev_indx = torch.zeros(ocr_max_num, char_max_num, dtype=torch.long)
    answer_mask = torch.zeros(ocr_max_num, char_max_num)
    if answer:
        ocr_char = 0 - torch.ones(ocr_max_num, char_max_num, dtype=torch.long)
    else:
        ocr_char = torch.zeros(ocr_max_num, char_max_num, dtype=torch.long)
    char_lable = torch.ones(ocr_max_num, dtype=torch.long)*(-1)
    _ocr_num = min(len(ocr_tokens), ocr_max_num)
    word_ids = torch.zeros(ocr_max_num, dtype=torch.long)
    for idx in range(_ocr_num):
        token = ocr_tokens[idx].lower().strip()
        ori_token = copy.deepcopy(token)
        randN = rng.random()
        if randN < adv_probability:  # self.adv_probability: #self.args.adv_probability:
            if ori_token.lower() in term2ids_dict:
                token = create_adv_word(token, rng)
            else:
                ori_token = find_related_word(token,term2ids_dict)
        if ori_token != token:
            if ori_token.lower() in term2ids_dict:
                char_lable[idx] = term2ids_dict[ori_token.lower()]
            else:
                char_lable[idx] = term2ids_dict['<unk>']
        tokenized = bert_tokenizer.encode(ori_token)
        if len(tokenized)>0:
            word_ids[idx] = torch.LongTensor(tokenized[:1])
            
        if ocr_tokens[idx] == '<pad>':
            if len(ocr_tokens) > 500:
                continue
            else:
                break
        if ocr_tokens[idx] in ['<s>', '</s>', '<unk>']:
            continue
        _char_num = min(len(ocr_tokens[idx]), char_max_num)
        c_i = -1
        for c_i in range(_char_num):
            c = ocr_tokens[idx][c_i]
            if c in char_set:
                c_idx = char_set[c]
            else:
                c_idx = char_set['<unk>']
            ocr_char[idx][c_i] = c_idx
            answer_mask[idx][c_i] = 1
            if answer:
                if c_i == 0:
                    prev_indx[idx][c_i] = 1
                else:
                    prev_indx[idx][c_i] = ocr_char[idx][c_i-1]
        if answer:
            c_i += 1
            c_idx = char_set['</s>']
            if c_i < char_max_num:
                ocr_char[idx][c_i] = c_idx
                answer_mask[idx][c_i] = 1
                if c_i == 0:
                    prev_indx[idx][c_i] = 1
                else:
                    prev_indx[idx][c_i] = ocr_char[idx][c_i-1]
    if answer:
        return ocr_char, answer_mask, prev_indx
    return ocr_char, answer_mask, char_lable, word_ids

def create_obj_mask(obj_tokens,obj_max_num):
    obj_masks = torch.zeros(obj_max_num,  dtype=torch.float)
    max_len= min(len(obj_tokens),obj_max_num)
    for i in range(max_len):
        if obj_tokens[i] != "background":
            obj_masks[i] = 1.0
    if torch.sum(obj_masks)==0:
        obj_masks[0]=1.0
    return obj_masks

def create_obj_labels(obj_tokens,obj_max_num):
    obj_labels = torch.ones(obj_max_num, obj_max_num, dtype=torch.float)*(-1.0)
    max_len= min(len(obj_tokens),obj_max_num)
    for i in range(max_len):
        if obj_tokens[i] == "background":
            continue
        for j in range(i+1,max_len):
            if obj_tokens[j]!="background":
                if obj_tokens[i].lower()==obj_tokens[j].lower():
                    obj_labels[i][j]=1.0
                    obj_labels[j][i] = 1.0
                else:
                    obj_labels[i][j] = 0.0
                    obj_labels[j][i] = 0.0
        obj_labels[i][i] = 1.0
    return obj_labels

def create_same_labels(ocr_tokens,ocr_max_num):
    ocr_labels = torch.ones(ocr_max_num, ocr_max_num, dtype=torch.float)*(-1.0)
    max_len= min(len(ocr_tokens),ocr_max_num)
    for i in range(max_len):
        for j in range(i+1,max_len):
            if ocr_tokens[i].lower()==ocr_tokens[j].lower():
                ocr_labels[i][j]=1.0
                ocr_labels[j][i] = 1.0
            else:
                ocr_labels[i][j] = 0.0
                ocr_labels[j][i] = 0.0
        ocr_labels[i][i] = 1.0
    return ocr_labels

def create_same_labels_id(word_id):
    ocr_max_num = len(word_id)
    ocr_labels = torch.zeros(ocr_max_num, ocr_max_num, dtype=torch.float)
    for i in range(ocr_max_num):
        if word_id[i] != 0 and word_id[i] != 101 and word_id[i] != 102:
            for j in range(i,ocr_max_num):
                if word_id[i]==word_id[j]:
                    ocr_labels[i][j]=1.0
                    ocr_labels[j][i] = 1.0
            ocr_labels[i][i] = 1.0
    return ocr_labels

def create_batch_labels(batch_labels):
    #batch_labels b*ocr_num*ocr_num
    batch_size = batch_labels.size(0)
    ocr_num = batch_labels.size(1)
    #labels = torch.arange(0, batch_size*ocr_num).to(device=logits_per_image.device)
    #result = torch.zeros(batch_size*ocr_num, batch_size*ocr_num, dtype=torch.long)
    result =torch.zeros(1, dtype=torch.float)
    for idx in range(batch_size):
        top = torch.zeros((idx*ocr_num, ocr_num), dtype=torch.float).to('cuda')
        middle = batch_labels[idx]
        bottom = torch.zeros(((batch_size-1-idx) * ocr_num, ocr_num), dtype=torch.float).to('cuda')
        row = torch.cat([top, middle],dim=0)
        row = torch.cat([row, bottom], dim=0)
        assert  row.size(0)==batch_size*ocr_num
        if result.size(0)==1:
            result = row
        else:
            result = torch.cat([result, row], dim=1)
    assert result.size(1) == batch_size * ocr_num
    for i in range(result.size(1)):
        if result[i,i] == -1:
            result[i,:] = -1
            result[:,i] = -1
    return result

def create_batch_labels_diag(batch_labels):
    #batch_labels b*ocr_num*ocr_num
    batch_size = batch_labels.size(0)
    ocr_num = batch_labels.size(1)
    #labels = torch.arange(0, batch_size*ocr_num).to(device=logits_per_image.device)
    #result = torch.zeros(batch_size*ocr_num, batch_size*ocr_num, dtype=torch.long)
    result =torch.zeros(1, dtype=torch.float)
    for idx in range(batch_size):
        top = torch.ones((idx*ocr_num, ocr_num), dtype=torch.float).to('cuda')*(-1)
        middle = batch_labels[idx]
        bottom = torch.ones(((batch_size-1-idx) * ocr_num, ocr_num), dtype=torch.float).to('cuda')*(-1)
        row = torch.cat([top, middle],dim=0)
        row = torch.cat([row, bottom], dim=0)
        assert  row.size(0)==batch_size*ocr_num
        if result.size(0)==1:
            result = row
        else:
            result = torch.cat([result, row], dim=1)
    assert result.size(1) == batch_size * ocr_num

    return result

