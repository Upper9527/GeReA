#!/usr/bin/env python
# coding: utf-8

import torch
import requests
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

from lavis.common.gradcam import getAttMap
from lavis.models import load_model_and_preprocess



# setup device to use
#device = 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model, vis_processors, txt_processors = load_model_and_preprocess(name="img2llm_okvqa", model_type="base", is_eval=True, device=device)

import json
import numpy
okvqa_image_id = []
okvqa = json.load(open("/home/hnu-a100/下载/cap_kvqa/okvqa_dataset/OpenEnded_mscoco_train2014_questions.json", "r"))

image_id_list = json.load(open("/home/hnu-a100/下载/cap_kvqa/okvqa_dataset/captions_train2014.json", "r"))

encoder_image_feature_dict = {}
saved_captions_dict = {}
saved_sorted_id_dict = {}
saved_sorted_examples_dict = {}
import time


count = 0
for sample in okvqa["questions"]:
    for image_id in image_id_list["images"]:
        if sample["image_id"] == image_id["id"]:
            image_path = "/home/hnu-a100/下载/cap_kvqa/okvqa_dataset/train2014/" + image_id["file_name"]
            break

    raw_image = Image.open(image_path).convert("RGB")
    start = time.time()

    question = sample["question"]
    print(question)

    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    question = txt_processors["eval"](question)

    samples = {"image": image, "text_input": [question]}


    samples = model.forward_itm(samples=samples)

    saved_captions = model.generate_instructblip(samples=samples)

    saved_captions_dict[str(sample['image_id'])+'<->'+str(sample['question_id'])] =saved_captions

    count = count + 1
    print(count)
    end = time.time()
    print(end - start)

with open('train2014_caption_data_instructblip_v12.json', 'w') as f:
    json.dump(saved_captions_dict, f, indent=2)

