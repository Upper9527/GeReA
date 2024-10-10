"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

"""

import random
import numpy as np
import spacy
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer

from lavis.common.dist_utils import download_cached_file
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam

open_pos = ["NOUN", "VERB", "ADJ", "ADV", "NUM"]



@registry.register_model("img2prompt_vqa")
class Img2PromptVQA(BaseModel):
    """
    Img2Prompt_VQA model consists of three submodels for zero-shot VQA:
        1. Image-questioning matching model
        2. Image captioning model
        3. Large Language model

    Supported model types:
        - base: BLIPITM, BLIPCaption, PNPUnifiedQAv2FiD (t5-base)
        - large: BLIPITM, BLIPCaption, PNPUnifiedQAv2FiD (t5-large)
        - 3b: BLIPITM, BLIPCaption, PNPUnifiedQAv2FiD (t5-3b)

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("img2prompt_vqa", "base", is_eval=True)
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/img2prompt-vqa/img2prompt_vqa_base.yaml",
    }

    def __init__(
        self,
        image_question_matching_model,
        image_captioning_model,
        question_generation_model,
        question_generation_tokenizer,
        offload_model=False,
    ):
        super().__init__()

        self.image_question_matching_model = image_question_matching_model
        self.image_captioning_model = image_captioning_model
        self.question_generation_model = question_generation_model
        self.question_generation_tokenizer = question_generation_tokenizer
        self.offload_model = offload_model
        self.nlp = spacy.load("en_core_web_sm")

    def forward_itm(self, samples, block_num=7):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
            block_num (int): The index of cross-attention block for gradcam computation.

        Returns:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
                - gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
        """
        image = samples["image"]

        question = [text.strip("?") for text in samples["text_input"]]
        tokenized_text = self.image_question_matching_model.tokenizer(
            question, padding="longest", truncation=True, return_tensors="pt"
        ).to(self.image_question_matching_model.device)
        with torch.set_grad_enabled(True):
            gradcams, _ = compute_gradcam(
                model=self.image_question_matching_model,
                visual_input=image,
                text_input=question,
                tokenized_text=tokenized_text,
                block_num=block_num,
            )

        gradcams = [gradcam_[1] for gradcam_ in gradcams]
        samples["gradcams"] = torch.stack(gradcams).reshape(
            samples["image"].size(0), -1
        )

        return samples

    def itm_rank(self, image_embeds, image_atts, encoder_input_ids, match_head="itm"):
        # breakpoint()
        encoder_input_ids = encoder_input_ids.clone()
        encoder_input_ids = encoder_input_ids[:, self.prompt_length - 1 :]
        text_attention_mask = (encoder_input_ids != self.tokenizer.pad_token_id).long()

        if match_head == "itm":
            # encoder_input_ids = encoder_input_ids.clone()
            encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
            output = self.text_encoder(
                encoder_input_ids,
                attention_mask=text_attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            itm_output = self.itm_head(output.last_hidden_state[:, 0, :])
            return itm_output  # , mask, token_length

        elif match_head == "itc":
            encoder_input_ids[:, 0] = self.tokenizer.cls_token_id
            text_output = self.text_encoder(
                encoder_input_ids,
                attention_mask=text_attention_mask,
                return_dict=True,
                mode="text",
            )
            image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
            text_feat = F.normalize(
                self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )

            sim = image_feat @ text_feat.t()
            return sim

    def forward_cap(
        self,
        samples,
        cap_max_length=128,
        cap_min_length=30,
        top_p=1,
        top_k=50,
        repetition_penalty=1.0,
        num_captions=25,
        num_patches=20,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
                - gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
            cap_max_length (int): The maximum length of the caption to be generated.
            cap_min_length (int): The minimum length of the caption to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            top_k (float): The number of the highest probability tokens for top-k sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions generated for each image.
            num_patches (int): Number of patches sampled for each image.

        Returns:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
                - gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
                - captions (nested list): A nested list of strings of total length batch_size * num_captions
        """
        encoder_out = self.image_captioning_model.forward_encoder(samples)
        # print(encoder_out)
        # print(encoder_out.size())
        # print(samples["image"].size())
        # print(samples['gradcams'].size())
        # exit()
        captions = [[] for _ in range(encoder_out.size(0))]
        # print("111")
        # print(captions)

        encoder_out_samples = []
        saved_encoder_out_samples = []
        # num = num_captions / 5
        for i in range(5):
            patch_id = (
                torch.multinomial(
                    samples["gradcams"].to(self.image_captioning_model.device),
                    num_patches,
                ).reshape(encoder_out.size(0), -1)
                #+ 1
            )
            # print("aa")
            # print(patch_id)
            # print(patch_id.size())
            # print("aaa")

            patch_id = (
                patch_id.sort(dim=1)
                .values.unsqueeze(-1)
                .expand(-1, -1, encoder_out.size(2))
            )
            # print("bb")
            # print(patch_id)
            # print(patch_id.size())
            # print("bbb")

            encoder_out_sample = torch.gather(encoder_out, 1, patch_id)
            np.save("pathch_id.npy", patch_id.cpu().detach().numpy())
            # print(patch_id)
            # print(patch_id.size())
            # print(encoder_out_sample)
            # print(encoder_out_sample.size())
            # exit()
            saved_encoder_out_samples.append(encoder_out_sample.detach().numpy())
            for j in range(5):
                encoder_out_samples.append(encoder_out_sample)

        # print("encoder_out_samples", len(encoder_out_samples))

        stacked = torch.stack(encoder_out_samples, dim=1)
        image_embeds = torch.flatten(
            stacked, start_dim=0, end_dim=1
        )  # (bsz*num_seq, num_patch, dim)

        # print(image_embeds.size())
        # print("awu")

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.image_captioning_model.device
        )
        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        import re

        def delete_boring_characters(sentence):
            return re.sub('[!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+', "", sentence)

        txt = delete_boring_characters(samples["text_input"][0].lower())
        # print(txt)
        # mydict = {"a": None, "night": None}
        # print(txt.translate(mydict))
        del_dist = ["here", "there", "much", "little", "very", "rather", "so", "too", "still" \
            , "quite", "perfectly", "enough", "extremely", "entirely", "almost", "slightly" \
            , "hardly", "how", "when", "where", "why", "what", "then", "i", "you", "he", "she" \
            , "they", "me", "you", "him", "her", "them", "my", "his", "your", "their", "hers" \
            , "this", "that", "those", "myself", "himself", "themselves", "which", "some", "many" \
            , "both", "whoever", "whomever", "a", "also", "and", "but", "whose", "whom", "how", "an" \
            , "are", "is", "the", "have", "has", "of", "", " ", "do", "does"]
        text = txt.split(" ")
        update_list = []
        for t in text:
            if t in del_dist:
                continue
            else:
                update_list.append(t)

        prompt_base = []
        prompt_base.append("Write down the facts that you know about this picture: ")
        prompt_base.append("Explain this picture based on the information below: " + ", ".join(update_list) + ". ")
        prompt_base.append("Explain this picture: ")
        prompt_base.append("Explain this picture according to the question: " + txt + ". ")
        prompt_base.append("Tell me something about the picture according to the question: " + txt + ". ")

        #prompt1 = ["Write down the facts that you know about this picture: ", "Explain this picture in as much detail as possible based on the information provided below: hairstyle blond called. ", " Explain this picture: ", "Explain this picture according to the question: what is the hairstyle of the blond called. ",  "Tell me something about the picture according to the question: what is the hairstyle of the blond called. "] * 5
        # print("lailelaodi1")
        # print(prompt1)
        # print("lailelaodi2")
        prompt1 = prompt_base * 5
        prompt = self.image_captioning_model.tokenizer(
            prompt1, return_tensors="pt"#, padding=True
        ).to(self.image_captioning_model.device)
        prompt.input_ids[:, 0] = self.image_captioning_model.tokenizer.bos_token_id
        prompt.input_ids = prompt.input_ids[:, :-1]

        decoder_out = self.image_captioning_model.text_decoder.generate(
            input_ids=prompt.input_ids,
            max_length=cap_max_length,
            min_length=cap_min_length,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=1,
            eos_token_id=self.image_captioning_model.tokenizer.sep_token_id,
            pad_token_id=self.image_captioning_model.tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            **model_kwargs
        )

        outputs = self.image_captioning_model.tokenizer.batch_decode(
            decoder_out, skip_special_tokens=True
        )

        saved_captions = []

        for counter, output in enumerate(outputs):
            caption = output[len(prompt1[counter]):]
            saved_captions.append(caption)
            # print(caption)
        # print(outputs)
        # print(len(outputs))
        # print("lalalal")

        samples["captions"] = saved_captions

        return samples, saved_encoder_out_samples, saved_captions

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=32,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        # print("here")
        encoder_out_samples = []

        with self.image_captioning_model.maybe_autocast():
            image_embeds = self.image_captioning_model.ln_vision(self.image_captioning_model.visual_encoder(image))

        patch_id = (
                torch.multinomial(
                    samples["gradcams"].to(image.device),
                    20,
                ).reshape(image_embeds.size(0), -1)
                + 1
        )

        patch_id = (
            patch_id.sort(dim=1)
            .values.unsqueeze(-1)
            .expand(-1, -1, image_embeds.size(2))
        )

        encoder_out_sample = torch.gather(image_embeds, 1, patch_id)

        encoder_out_samples.append(encoder_out_sample)

        stacked = torch.stack(encoder_out_samples, dim=1)
        image_embeds = torch.flatten(
            stacked, start_dim=0, end_dim=1
        )  # (bsz*num_seq, num_patch, dim)

        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.image_captioning_model.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.image_captioning_model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_t5 = self.image_captioning_model.t5_proj(query_output.last_hidden_state)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = self.image_captioning_model.prompt

        if isinstance(prompt, str):
            prompt = [prompt] * image.size(0)
        else:
            assert len(prompt) == image.size(
                0
            ), "The number of prompts must be equal to the batch size."

        input_tokens = self.image_captioning_model.t5_tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        with self.image_captioning_model.maybe_autocast(dtype=torch.bfloat16):
            inputs_embeds = self.image_captioning_model.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

            outputs = self.image_captioning_model.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )
            output_text = self.image_captioning_model.t5_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        return output_text

    def answer_extraction(self, caption, num_question_generation=30):
        cap_use = ""
        # print(caption)
        caption = caption
        ans_to_cap_dict = {}
        answers = []
        for cap_idx, cap in enumerate(caption):
            # print(cap)
            cap_use += cap
            cap = cap.strip().strip(".")
            # print(cap)
            cap = self.nlp(cap)
            for token in cap:  # Noun /Verb/Adj//NUM
                if token.pos_ in open_pos:
                    if token.text.lower() not in ans_to_cap_dict:
                        ans_to_cap_dict[token.text.lower()] = [cap_idx]
                    else:
                        if cap_idx not in ans_to_cap_dict[token.text.lower()]:
                            ans_to_cap_dict[token.text.lower()].append(cap_idx)
                    answers.append(token.text)
            for ent in cap.ents:

                if ent.text not in answers:
                    if ent.text.lower() not in ans_to_cap_dict:
                        ans_to_cap_dict[ent.text.lower()] = [cap_idx]
                    else:
                        if cap_idx not in ans_to_cap_dict[ent.text.lower()]:
                            ans_to_cap_dict[ent.text.lower()].append(cap_idx)
                    answers.append(ent.text)
            for chunk in cap.noun_chunks:
                if len(chunk.text.split()) < 4:
                    if chunk.text.lower() not in ans_to_cap_dict:
                        ans_to_cap_dict[chunk.text.lower()] = [cap_idx]
                    else:
                        if cap_idx not in ans_to_cap_dict[chunk.text.lower()]:
                            ans_to_cap_dict[chunk.text.lower()].append(cap_idx)
                    #                 print(chunk.text)
                    answers.append(chunk.text)
        answers = sorted(answers, key=answers.count, reverse=True)
        real_answers = []
        for i in answers:
            i = i + "."
            if i not in real_answers:
                real_answers.append(i)

        contexts_for_question_generation = []
        answers = []
        for ans in real_answers[
            :num_question_generation
        ]:  # Generate questions for 30 answers with max frequencies.
            contexts_for_question_generation.append(
                "answer: %s  context: %s." % (ans, cap_use)
            )
            answers.append(ans)
        contexts_for_question_generation.append(
            "answer: %s  context: %s." % ("yes.", cap_use)
        )
        answers.append("yes.")
        return contexts_for_question_generation, answers, ans_to_cap_dict

    def forward_qa_generation(self, samples):
        caption = samples["captions"][0]
        (
            contexts_for_question_generation,
            answers,
            ans_to_cap_dict,
        ) = self.answer_extraction(caption)
        inputs = self.question_generation_tokenizer(
            contexts_for_question_generation,
            padding="longest",
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        ).to(self.device)
        question_size = inputs.input_ids.shape[0]
        cur_b = 0
        true_input_size = 10
        outputs_list = []
        while cur_b < question_size:
            outputs = self.question_generation_model.generate(
                input_ids=inputs.input_ids[cur_b : cur_b + true_input_size],
                attention_mask=inputs.attention_mask[cur_b : cur_b + true_input_size],
                num_beams=3,
                max_length=30,
            )
            questions = self.question_generation_tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            outputs_list += questions
            cur_b += true_input_size
        questions = outputs_list
        samples["questions"] = questions
        samples["answers"] = answers
        samples["ans_to_cap_dict"] = ans_to_cap_dict
        # results.append({"question_id": ques_id, "question":questions,"answer":answers})
        return samples

    def create_context_prompt(self, samples, num_caps_per_img=30):
        ans_dict_queid = samples["ans_to_cap_dict"]
        # print(ans_dict_queid)
        caption = samples["captions"][0]
        answers = samples["answers"]
        Context_Prompt = ""
        mycontexts_id = []
        for idx in range(num_caps_per_img):
            cap_id_list = ans_dict_queid.get(
                answers[(len(answers) - 1 - idx) % len(answers)][:-1].lower(), [0]
            )
            for cap_id in cap_id_list:
                if cap_id not in mycontexts_id:
                    Context_Prompt += caption[cap_id]
                    mycontexts_id.append(cap_id)
                    break  # We just take one cap for each answer
        samples["Context_Prompt"] = Context_Prompt
        return Context_Prompt

    def create_task_prompt(
        self, samples, question_type="neural", num_question_per_img=30
    ):
        syn_question_queid = samples["questions"]
        syn_ans_queid = samples["answers"]
        Task_Prompt = ""
        for idx in range(num_question_per_img):
            # if config['random_question']:
            #     qa_idx = random.randint(0, len(syn_question_queid) - 1)
            # else:
            qa_idx = idx
            if (
                question_type != "rule" and num_question_per_img > 0 and idx < 1
            ):  ## yes and no questions for vqav2
                # Task_Prompt += "Question:"
                # Task_Prompt += syn_question_queid_next[-1]
                # Task_Prompt += '\n'
                # Task_Prompt += "Answer:no\n"
                Task_Prompt += "Question:"
                Task_Prompt += syn_question_queid[-1]
                Task_Prompt += "\n"
                Task_Prompt += "Answer:"
                Task_Prompt += "yes\n"
                Task_Prompt += "Question:Is this a toilet?\n"
                Task_Prompt += "Answer:no\n"
            if "question_type" == "rule":  # Rule-Based Question Generation
                Noun_Questions = [
                    "What item is this in this picture?",
                    "What item is that in this picture?",
                ]

                Verb_Questions = [
                    "What action is being done in this picture?",
                    "Why is this item doing in this picture?",
                    "Which action is being taken in this picture?",
                    "What action is item doing in this picture?",
                    "What action is item performing in this picture?",
                ]

                Adj_Questions = [
                    "How to describe one item in this picture?",
                    "What is item's ADJ TYPE in this picture?",
                    "What is the ADJ TYPE in this picture?",
                ]

                Task_Prompt += "Question:"
                doc = self.nlp(syn_ans_queid[(qa_idx) % len(syn_ans_queid)][:-1].lower())
                if doc[-1].pos_ == "NOUN":
                    Task_Prompt += Noun_Questions[
                        random.randint(0, len(Noun_Questions) - 1)
                    ]
                elif doc[-1].pos_ == "VERB":
                    Task_Prompt += Verb_Questions[
                        random.randint(0, len(Verb_Questions) - 1)
                    ]
                elif doc[-1].pos_ == "ADJ":
                    Task_Prompt += Adj_Questions[
                        random.randint(0, len(Adj_Questions) - 1)
                    ]

                Task_Prompt += "\n"

                Task_Prompt += "Answer:"
                Task_Prompt += syn_ans_queid[(qa_idx) % len(syn_ans_queid)][:-1].lower()
                Task_Prompt += "\n"
        samples["Task_Prompt"] = Task_Prompt
        # print(Task_Prompt)
        return Task_Prompt

    def prompts_construction(
        self,
        samples,
        question_type="neural",
        num_caps_per_img=30,
        num_question_per_img=30,
    ):
        Prompt = "Please reason the answer of the questions according to the given contexts.\n"

        Context_Prompt = self.create_context_prompt(samples, num_caps_per_img)

        Task_Prompt = self.create_task_prompt(
            samples, question_type, num_question_per_img
        )

        Img2Prompt = (
            Prompt
            + "Contexts:"
            + Context_Prompt
            + "\n"
            + Task_Prompt
            + "Question:"
            + samples["text_input"][0]
            + "\nAnswer:"
        )
        return Img2Prompt

    def prepare_LLM_input(
        self,
        samples,
        num_beams=1,
        inference_method="generate",
        max_len=20,
        min_len=0,
        internal_bsz_fid=1,
        num_captions=50,
        num_captions_fid=1,
        cap_max_length=20,
        cap_min_length=10,
        top_k=50,
        top_p=1,
        repetition_penalty=1,
        num_patches=20,
        block_num=7,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). Default H=480, W=480.
                - text_input (str or [str]): String or a list of strings, each string is a question.
                                             The number of questions must be equal to the batch size. If a single string, will be converted to a list of string, with length 1 first.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            inference_method (str): Inference method. Must be "generate". The model will generate answers.
            max_len (int): Maximum length of generated answers.
            min_len (int): Minimum length of generated answers.
            internal_bsz_fid (int): Internal batch size when using FiD decoding.
            num_captions (int): Number of captions generated for each image.
            num_captions_fid (int): Number of captions concatenated with a question during FiD decoding.
            cap_max_length (int): The maximum length of the caption to be generated.
            cap_min_length (int): The minimum length of the caption to be generated.
            top_k (float): The number of the highest probability tokens for top-k sampling.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_patches (int): Number of patches sampled for each image.
            block_num (int): The index of cross-attention block for gradcam computation.

        Returns:
            List: A list of strings, each string is an answer.
            gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
            captions (nested list): A nested list of strings of total length batch_size * num_captions
        """
        assert inference_method in [
            "generate",
        ], "Inference method must be 'generate', got {}.".format(inference_method)

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        assert len(samples["text_input"]) == samples["image"].size(
            0
        ), "The number of questions must be equal to the batch size."

        samples = self.forward_itm(samples, block_num=block_num)

        samples = self.forward_cap(
            samples,
            cap_max_length=cap_max_length,
            cap_min_length=cap_min_length,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_captions=num_captions,
            num_patches=num_patches,
        )

        if self.offload_model:
            samples["image"] = samples["image"].to("cpu")
            self.image_question_matching_model.to("cpu")
            self.image_captioning_model.to("cpu")
        torch.cuda.empty_cache()

        pred_answers = self.forward_qa(
            samples,
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            internal_bsz_fid=internal_bsz_fid,
            num_captions=num_captions,
            num_captions_fid=num_captions_fid,
        )

        if self.offload_model:
            self.image_question_matching_model.to(self.question_answering_model.device)
            self.image_captioning_model.to(self.question_answering_model.device)

        return pred_answers, samples["captions"], samples["gradcams"]

    @classmethod
    def from_config(cls, model_config):
        itm_config = model_config.image_question_matching_model
        cap_config = model_config.image_captioning_model

        itm_cls = registry.get_model_class(itm_config.arch)
        cap_cls = registry.get_model_class(cap_config.arch)

        image_question_matching_model = itm_cls.from_config(itm_config)
        image_captioning_model = cap_cls.from_config(cap_config)

        question_generation_tokenizer = T5Tokenizer.from_pretrained(
            "google/t5-large-lm-adapt"
        )
        question_generation_model = T5ForConditionalGeneration.from_pretrained(
            "google/t5-large-lm-adapt"
        )
        cached_file = download_cached_file(
            "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/projects/img2prompt/T5_large_QG.pth",
            check_hash=False,
            progress=True,
        )
        checkpoint = torch.load(cached_file, map_location="cpu")
        state_dict = checkpoint["model"]
        question_generation_model.load_state_dict(state_dict)
        model = cls(
            image_question_matching_model=image_question_matching_model,
            image_captioning_model=image_captioning_model,
            question_generation_model=question_generation_model,
            question_generation_tokenizer=question_generation_tokenizer,
            offload_model=False,
        )

        return model


@registry.register_model("img2llm_okvqa")
class Img2LlmOKVQA(BaseModel):
    """
    Img2Prompt_VQA model consists of three submodels for zero-shot VQA:
        1. Image-questioning matching model
        2. Image captioning model
        3. Large Language model

    Supported model types:
        - base: BLIPITM, BLIPCaption, PNPUnifiedQAv2FiD (t5-base)
        - large: BLIPITM, BLIPCaption, PNPUnifiedQAv2FiD (t5-large)
        - 3b: BLIPITM, BLIPCaption, PNPUnifiedQAv2FiD (t5-3b)

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("img2prompt_vqa", "base", is_eval=True)
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/img2prompt-vqa/img2llm_okvqa_base_instructblip.yaml",
    }

    def __init__(
            self,
            image_question_matching_model,
            image_captioning_model,
            offload_model=False,
    ):
        super().__init__()

        self.image_question_matching_model = image_question_matching_model
        self.image_captioning_model = image_captioning_model
        self.offload_model = offload_model
        self.nlp = spacy.load("en_core_web_sm")

    def forward_itm(self, samples, block_num=7):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
            block_num (int): The index of cross-attention block for gradcam computation.

        Returns:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
                - gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
        """
        image = samples["image"]
        question = [text.strip("?") for text in samples["text_input"]]
        tokenized_text = self.image_question_matching_model.tokenizer(
            question, padding="longest", truncation=True, return_tensors="pt"
        ).to(self.image_question_matching_model.device)
        with torch.set_grad_enabled(True):
            gradcams, _ = compute_gradcam(
                model=self.image_question_matching_model,
                visual_input=image,
                text_input=question,
                tokenized_text=tokenized_text,
                block_num=block_num,
            )

        gradcams = [gradcam_[1] for gradcam_ in gradcams]
        samples["gradcams"] = torch.stack(gradcams).reshape(
            samples["image"].size(0), -1
        )

        return samples

    def itm_rank(self, image_embeds, image_atts, encoder_input_ids, match_head="itm"):
        # breakpoint()
        encoder_input_ids = encoder_input_ids.clone()
        encoder_input_ids = encoder_input_ids[:, self.prompt_length - 1:]
        text_attention_mask = (encoder_input_ids != self.tokenizer.pad_token_id).long()

        if match_head == "itm":
            # encoder_input_ids = encoder_input_ids.clone()
            encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
            output = self.text_encoder(
                encoder_input_ids,
                attention_mask=text_attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            itm_output = self.itm_head(output.last_hidden_state[:, 0, :])
            return itm_output  # , mask, token_length

        elif match_head == "itc":
            encoder_input_ids[:, 0] = self.tokenizer.cls_token_id
            text_output = self.text_encoder(
                encoder_input_ids,
                attention_mask=text_attention_mask,
                return_dict=True,
                mode="text",
            )
            image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
            text_feat = F.normalize(
                self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )

            sim = image_feat @ text_feat.t()
            return sim

    @torch.no_grad()
    def generate_instructblip(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=25,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1,
        num_captions=1,
        temperature=1,
    ):
        self.image_captioning_model.llm_tokenizer.padding_side = "left"

        all_output = []
        import re
        def delete_boring_characters(sentence):
            return re.sub('[!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+', "", sentence)

        txt = delete_boring_characters(samples["text_input"][0].lower())

        del_dist = ["here", "there", "much", "little", "very", "rather", "so", "too", "still" \
            , "quite", "perfectly", "enough", "extremely", "entirely", "almost", "slightly" \
            , "hardly", "how", "when", "where", "why", "what", "then", "i", "you", "he", "she" \
            , "they", "me", "you", "him", "her", "them", "my", "his", "your", "their", "hers" \
            , "this", "that", "those", "myself", "himself", "themselves", "which", "some", "many" \
            , "both", "whoever", "whomever", "a", "also", "and", "but", "whose", "whom", "how", "an" \
            , "are", "is", "the", "have", "has", "of", "", " ", "do", "does", "was", "could", "would", "be"
            , "did", "or", "it", "its", "can", "of"]
        text = txt.split(" ")
        update_list = []
        for t in text:
            if t in del_dist:
                continue
            else:
                update_list.append(t)

        prompt_base = []
        prompt_base.append("Write down the facts that you know about this picture.")
        prompt_base.append(
            "Explain this picture in as much detail as much possible based on the information provided below: " + ", ".join(
                update_list) + ".")
        prompt_base.append(txt + "?")
        prompt_base.append("question:" + txt + "? the answer: ")
        prompt_base.append("question:" + txt + "? according to the question and image, we know that ")
        prompt_base.append("Explain this picture according to the question: " + txt + "?")


        image = samples["image"]

        with self.image_captioning_model.maybe_autocast():
            image_embeds = self.image_captioning_model.ln_vision(self.image_captioning_model.visual_encoder(image))

        # encoder_out_samples = []
        for num_patch in range(15):
            encoder_out_samples = []
            patch_id = (
                    torch.multinomial(
                        samples["gradcams"].to(image.device),
                        20,
                    ).reshape(image_embeds.size(0), -1)
                    #+ 1
            )

            patch_id = (
                patch_id.sort(dim=1)
                .values.unsqueeze(-1)
                .expand(-1, -1, image_embeds.size(2))
            )

            encoder_out_sample = torch.gather(image_embeds, 1, patch_id)

            encoder_out_samples.append(encoder_out_sample)

            stacked = torch.stack(encoder_out_samples, dim=1)
            image_embeds1 = torch.flatten(
                stacked, start_dim=0, end_dim=1
            )  # (bsz*num_seq, num_patch, dim)

            image_embeds1 = image_embeds1.float().to(image.device)

            image_atts1 = torch.ones(image_embeds1.size()[:-1], dtype=torch.long).to(image.device)

            for k in range(0, 6):
                query_tokens = self.image_captioning_model.query_tokens.expand(1, -1, -1)
                prompt = prompt_base[k]
                if self.image_captioning_model.qformer_text_input:
                    text_Qformer = self.image_captioning_model.tokenizer(
                        prompt,
                        padding='longest',
                        truncation=True,
                        max_length=self.image_captioning_model.max_txt_len,
                        return_tensors="pt",
                    ).to(image.device)
                    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
                    Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask], dim=1)

            # image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

                if self.image_captioning_model.qformer_text_input:
                    query_output = self.image_captioning_model.Qformer.bert(
                        text_Qformer.input_ids,
                        attention_mask=Qformer_atts,
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds1,
                        encoder_attention_mask=image_atts1,
                        return_dict=True,
                    )
                else:
                    query_output = self.image_captioning_model.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds1,
                        encoder_attention_mask=image_atts1,
                        return_dict=True,
                    )

                inputs_llm = self.image_captioning_model.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
                atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

                llm_tokens = self.image_captioning_model.llm_tokenizer(
                    prompt,
                    padding="longest",
                    return_tensors="pt"
                ).to(image.device)

                with self.image_captioning_model.maybe_autocast():
                    inputs_embeds = self.image_captioning_model.llm_model.get_input_embeddings()(llm_tokens.input_ids)
                    inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
                    attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask], dim=1)

                    outputs = self.image_captioning_model.llm_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        do_sample=use_nucleus_sampling,
                        top_p=top_p,
                        temperature=temperature,
                        num_beams=num_beams,
                        max_length=max_length,
                        min_length=min_length,
                        # eos_token_id=self.eos_token_id,
                        repetition_penalty=repetition_penalty,
                        length_penalty=length_penalty,
                        num_return_sequences=num_captions,
                    )


                outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
                output_text = self.image_captioning_model.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                output_text = [text.strip() for text in output_text]
                all_output = all_output + output_text

        return all_output


    @classmethod
    def from_config(cls, model_config):
        itm_config = model_config.image_question_matching_model
        cap_config = model_config.image_captioning_model

        itm_cls = registry.get_model_class(itm_config.arch)
        cap_cls = registry.get_model_class(cap_config.arch)

        image_question_matching_model = itm_cls.from_config(itm_config)
        image_captioning_model = cap_cls.from_config(cap_config)

        question_generation_tokenizer = T5Tokenizer.from_pretrained(
            "google/t5-large-lm-adapt"
        )
        question_generation_model = T5ForConditionalGeneration.from_pretrained(
            "google/t5-large-lm-adapt"
        )
        cached_file = download_cached_file(
            "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/projects/img2prompt/T5_large_QG.pth",
            check_hash=False,
            progress=True,
        )
        checkpoint = torch.load(cached_file, map_location="cpu")
        state_dict = checkpoint["model"]
        question_generation_model.load_state_dict(state_dict)
        model = cls(
            image_question_matching_model=image_question_matching_model,
            image_captioning_model=image_captioning_model,
            offload_model=False,
        )

        return model


