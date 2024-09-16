
import json


okvqa = json.load(open("../okvqa_dataset/mscoco_val2014_annotations.json", "r"))

gold_answer_dict = {}

for annotations in okvqa["annotations"]:
    answer_list = []
    for gold_answer in annotations["answers"]:
        answer_list.append(gold_answer["answer"])
    gold_answer_dict[str(annotations['image_id'])+'<->'+str(annotations['question_id'])] = answer_list


guided_caption_dict = json.load(
            open("../okvqa_dataset/val2014_captions_dict_blip2.json", "r"))


guided_caption_dict_blip = json.load(
            open("../okvqa_dataset/val2014v6_saved_sorted_examples_dict.json", "r"))

count = 0



for key, captions in guided_caption_dict.items():
    flag = 0
    num = []
    # for j in captions[:10]:
    #     num = num + j.split(" ")
    # print(num)
    # inter_caption = []
    # for i in captions:
    #     inter_caption = inter_caption + i.split(" ")
    #     inter_caption.append(".")
    # saved_caption = " ".join(inter_caption[:160])
    # if saved_caption[-1] != '.':
    #     saved_caption += '.'
    for answer in gold_answer_dict[key]:
        if flag == 1:
            break
        for caption in captions:
            if answer in caption:
                count = count + 1
                flag = 1
                break
        if flag == 0:
            for k in guided_caption_dict_blip[key][:70]:
                if answer in k:
                    count = count + 1
                    flag = 1
                    break


# for key in gold_answer_dict:
#     flag = 0
#     for answer in gold_answer_dict[key]:
#         if flag == 1:
#             break
#         for caption in guided_caption_dict[key]:
#             if answer in caption:
#                 count = count + 1
#                 flag = 1
#                 break

print(count/5046)
