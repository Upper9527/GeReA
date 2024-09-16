import pdb
import torch
import random
import json
import numpy as np
from collections import Counter

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 opt,
                 data,
                 context_prefix='context:',
                 question_prefix='question:',
                 em_title_prefix='entity:',
                 em_passage_prefix='description:',
                 im_title_prefix='candidate:',
                 im_passage_prefix='evidence:'):
        self.data = data
        self.n_box = opt.box_number
        self.context_prefix = context_prefix
        self.n_im_context = opt.n_im_context
        self.n_ex_context = opt.n_ex_context
        self.question_prefix = question_prefix
        self.em_title_prefix = em_title_prefix
        self.em_passage_prefix = em_passage_prefix
        self.im_title_prefix = im_title_prefix
        self.im_passage_prefix = im_passage_prefix
        self.n_tags = opt.n_tags

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        # 统计list中每个元素出现的个数
        eleCounts = Counter(example['answers_list'])
        # most_common()返回出现次数排名前n个的元素，不输入时默认按照出现次数对所有数据排序
        top_one = eleCounts.most_common()
        true_answer = []
        for sample in top_one:
            if sample[1] >= 3:
                true_answer.append(sample[0])
        if 'target' in example:
            target = example['target']
            return target + ' </s>'
        elif 'answers_list' in example:
            if true_answer == []:
                return random.choice(example['answers_list']) + ' </s>'
            else:
                return random.choice(example['answers_list']) + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']

        target = self.get_target(example)

        vision_feature = np.zeros(example["image_feature"].shape,dtype=np.float32)

        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'vis_feat': example["image_feature"],
            'passages' : example["in_context"]
        }


    def get_example(self, index):
        return self.data[index]

    def extract_im_element(self, im_context):
        im_context_new, entities = [], []
        for i in im_context:
            key = i['title']
            if key not in entities:
                entities.append(key)
                im_context_new.append(i)

        return im_context_new

def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        
        vis_feats = [torch.unsqueeze(torch.from_numpy(ex['vis_feat']), 0) for ex in batch]
        # pos = [torch.unsqueeze(torch.from_numpy(ex['pos']), 0) for ex in batch]
        vis_feats = torch.cat(vis_feats, dim=0)
        # pos = torch.cat(pos, dim=0)
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example):
            if example['passages'] is None:
                return [example['question']]
            return [t for t in example['passages']]
        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)

        return (index, target_ids, target_mask, passage_ids, passage_masks, vis_feats)

def load_data(data_path=None, split_type="val2014", global_rank=-1, world_size=-1):
    okvqa_image_id = []
    okvqa_train_answer_dict = {}
    okvqa_train_question_dict = {}
    okvqa_similar = json.load(open("./okvqa_dataset/mscoco_train2014_annotations.json", "r"))
    for annotations in okvqa_similar["annotations"]:
        okvqa_train_answer_dict[str(annotations["question_id"])] = [answer["answer"] for answer in annotations["answers"]]


    question_for_simliar = json.load(
        open('./okvqa_dataset/OpenEnded_mscoco_train2014_questions.json', 'r'))

    for questions in question_for_simliar["questions"]:
        okvqa_train_question_dict[str(questions["question_id"])] = questions["question"]

    guided_caption_dict_for_complete = json.load(
        open("./okvqa_dataset/okvqa_train2014_caption_llava_patch_id_30_0_9009_10_32.json", "r"))
    guided_caption_dict_blip2_similar = json.load(
        open("./okvqa_dataset/train2014_caption_data_instructblip_v12.json", "r"))

    caption_dict_for_similar = {}
    for key, caption in guided_caption_dict_for_complete.items():
        id = str(key.split('<->')[1])
        caption_dict_for_similar[id] = guided_caption_dict_for_complete[key][:80]+ guided_caption_dict_blip2_similar[key][:40]

    if split_type == "val2014":
        okvqa = json.load(open("./okvqa_dataset/mscoco_val2014_annotations.json", "r"))
        for annotations in okvqa["annotations"]:
            okvqa_image_id.append(annotations["image_id"])
        image_id_list = json.load(open("./okvqa_dataset/captions_val2014.json", "r"))
    else:
        okvqa = json.load(open("./okvqa_dataset/mscoco_train2014_annotations.json", "r"))
        for annotations in okvqa["annotations"]:
            okvqa_image_id.append(annotations["image_id"])
        image_id_list = json.load(open("./okvqa_dataset/captions_train2014.json", "r"))

    image2imageid = {}
    for image_id in image_id_list["images"]:
        if image_id["id"] not in okvqa_image_id:
            continue
        else:
            image2imageid[image_id["file_name"]] = image_id["id"]
    if split_type == "val2014":
        question_anno = json.load(
            open('./okvqa_dataset/OpenEnded_mscoco_val2014_questions.json', 'r'))
        guided_caption_dict = json.load(
            open("./okvqa_dataset/okvqa_val2014_caption_llava_patch_id_30_0_5046_10_32.json", "r"))
        guided_caption_dict_blip2 = json.load(
            open("./okvqa_dataset/val2014_caption_data_instructblip_v12.json", "r"))
        image_features = np.load('./okvqa_dataset/detr_encoded_val2014_dic.npy', allow_pickle=True).item()
        similar_qids = json.load(
            open("./okvqa_dataset/similar_val_qids.json", "r"))
    else:
        question_anno = json.load(
            open('./okvqa_dataset/OpenEnded_mscoco_train2014_questions.json', 'r'))
        guided_caption_dict = json.load(
            open("./okvqa_dataset/okvqa_train2014_caption_llava_patch_id_30_0_9009_10_32.json", "r"))
        guided_caption_dict_blip2 = json.load(
            open("./okvqa_dataset/train2014_caption_data_instructblip_v12.json", "r"))
        image_features = np.load('./okvqa_dataset/detr_encoded_train2014_dic.npy',
                                 allow_pickle=True).item()
        similar_qids = json.load(
            open("./okvqa_dataset/similar_train_qids.json", "r"))
        for key, similar_qid in similar_qids.items():
            similar_qids[key] = similar_qid[1:]

    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    else:
        data = np.load(data_path, allow_pickle=True)
    examples = []
    in_context_examples = []

    for k, example in enumerate(data):
        if global_rank > -1 and not k % world_size == global_rank:
            continue
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        
        imageid = image2imageid[example["image_name"]]
        answer_candidate = []

        for sample in question_anno["questions"]:
            if sample["image_id"] == imageid and sample["question"] == example['question']:
                example["guided_caption"] =guided_caption_dict[str(sample['image_id'])+'<->'+str(sample['question_id'])][:80]+ guided_caption_dict_blip2[str(sample['image_id'])+'<->'+str(sample['question_id'])][:40]

                line_prefix = "===\n"
                in_context_list = []
                for index, prompt_caption in enumerate(example["guided_caption"]):
                    prompt_text = ""
                    aa = similar_qids[str(sample["question_id"])][0:5]
                    random.shuffle(aa)
                    for qid in aa:
                        qid = str(qid)
                        prompt_text += line_prefix + f'Context: {caption_dict_for_similar[qid][index]}\n'
                        prompt_text += line_prefix + f'Question: {okvqa_train_question_dict[qid]}\n'
                        prompt_text += line_prefix + f'Answer: {random.choice(okvqa_train_answer_dict[qid])}\n'
                        prompt_text += '\n\n'
                    prompt_text += line_prefix + f'Context: {prompt_caption}\n'
                    prompt_text += line_prefix + f'Question: {example["question"]}\n'
                    prompt_text += line_prefix + f'Answer: \n'
                    in_context_list.append(prompt_text)
                example["in_context"] = in_context_list
                example["image_feature"] = image_features[int(sample['image_id'])].squeeze(0)

        examples.append(example)
        in_context_examples.append(example["in_context"])

    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples


class RetrieverCollator(object):
    def __init__(self, tokenizer, passage_maxlength=200, question_maxlength=40):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])

        question = [ex['question'] for ex in batch]
        question = self.tokenizer.batch_encode_plus(
            question,
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.question_maxlength,
            truncation=True
        )
        question_ids = question['input_ids']
        question_mask = question['attention_mask'].bool()

        if batch[0]['scores'] is None or batch[0]['passages'] is None:
            return index, question_ids, question_mask, None, None, None

        scores = [ex['scores'] for ex in batch]
        scores = torch.stack(scores, dim=0)

        passages = [ex['passages'] for ex in batch]
        passage_ids, passage_masks = encode_passages(
            passages,
            self.tokenizer,
            self.passage_maxlength
        )

        return (index, question_ids, question_mask, passage_ids, passage_masks, scores)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.data = data
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        text = self.title_prefix + " " + example[2] + " " + \
            self.passage_prefix + " " + example[1]
        return example[0], text

class TextCollator(object):
    def __init__(self, tokenizer, maxlength=200):
        self.tokenizer = tokenizer
        self.maxlength = maxlength

    def __call__(self, batch):
        index = [x[0] for x in batch]
        encoded_batch = self.tokenizer.batch_encode_plus(
            [x[1] for x in batch],
            pad_to_max_length=True,
            return_tensors="pt",
            max_length=self.maxlength,
            truncation=True
        )
        text_ids = encoded_batch['input_ids']
        text_mask = encoded_batch['attention_mask'].bool()

        return index, text_ids, text_mask
