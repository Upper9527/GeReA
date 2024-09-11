from argparse import ArgumentParser
import os
import pickle
import json
import pdb
import string
import regex
import argparse

def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def okvqa_ems(prediction, ground_truths):
    correct_num = 0
    for gt in ground_truths:
        correct_num += exact_match_score(prediction, gt)
    cur_acc = min(float(correct_num/3), 1.0)
    return cur_acc


def load_json(file_path):
    with open(file_path, 'r') as input_file:
        data = json.load(input_file)
    return data

'''
   load predictions from our model
   input: the path to our prediction file
   output: a dictionary where key is image_name#question_id and value is the prediction
'''
def load_predictions(file_path, gold_answers):
    predictions = {}
    m = json.load(open(file_path)) 

    count = 0
    for img_que_id in gold_answers:
        predictions[img_que_id] = m[count]['answer']
        count += 1
    return predictions


'''
    load groundtruth from mscoco_val2014_annotations.json
    input: the path to mascoco_val2014_annotations.json
    output: a dictionary where key is image_name#question_id and value is a list of gold_answers
'''
def load_gt(file_path):
    gold_answers = {}
    gt_data = load_json(file_path)
    for annotation in gt_data['annotations']:
        img_id = annotation['image_id']
        question_id = annotation['question_id']
        answers = annotation['answers']
        gt_answer = [answer_info['answer'] for answer_info in answers]
        img_key = 'COCO_val2014_{}.jpg#{}'.format(str(img_id).zfill(12), question_id)
        gold_answers[img_key] = gt_answer
    return gold_answers


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--pred_path", default="/mnt/2030154A30152874/ziyu/knowledge-based-VQA/LLaVA-main/okvqa_val2014_zero_shot_performance_llava_0827.json", type=str, help="predicion file path")
    parser.add_argument("--gt_path", default="eval/mscoco_val2014_annotations.json", type=str, help="gt file path")
    args = parser.parse_args()

    pred_path, gt_path = args.pred_path, args.gt_path

    # Load gold_answers and predctions
    gold_answers = load_gt(gt_path)
    predictions = load_predictions(pred_path, gold_answers)

    acc = []

    # Iterate over image_name#question_ids, 5046 in total
    for img_que_id in predictions:
        prediction, gold_answer = predictions[img_que_id], gold_answers[img_que_id]
        # As our model is a generative model, we use an official implementation of exact match between predictions and gold_answers
        # It will normalize the output text before comparison as SQuAD evaluation.
        cur_acc = okvqa_ems(prediction, gold_answer)
        acc.append(cur_acc)

    # Calculate the average accuracy
    average_acc = sum(acc)/len(acc)

    print('Average accuracy is: {}'.format(average_acc))
