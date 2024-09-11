# Official Code of "GeReA: Question-Aware Prompt Captions for Knowledge-based Visual Question Answering"

# !!!!!Repo Under Construction!!!!
![GeReA](https://github.com/Upper9527/GeReA/blob/main/figs/framework.png)


**GeReA** is a method for knowledge-based VQA task, as described in ([link to paper](https://arxiv.org/abs/2402.02503)).
>Knowledge-based visual question answering (VQA) requires world knowledge beyond the image for accurate answer. Recently, instead of extra knowledge bases, a large language model (LLM) like GPT-3 is activated as an implicit knowledge engine to jointly acquire and reason the necessary knowledge for answering by converting images into textual information (e.g., captions and answer candidates). However, such conversion may introduce irrelevant information, which causes the LLM to misinterpret images and ignore visual details crucial for accurate knowledge. We argue that multimodal large language model (MLLM) is a better implicit knowledge engine than the LLM for its superior capability of visual understanding. Despite this, how to activate the capacity of MLLM as the implicit knowledge engine has not been explored yet. Therefore, we propose GeReA, a generate-reason framework that prompts a MLLM like InstructBLIP with question relevant vision and language information to generate knowledge-relevant descriptions and reasons those descriptions for knowledge-based VQA. Specifically, the question-relevant image regions and question-specific manual prompts are encoded in the MLLM to generate the knowledge relevant descriptions, referred to as question-aware prompt captions. After that, the question-aware prompt captions, image-question pair, and similar samples are sent into the multi-modal reasoning model to learn a joint knowledge-image-question representation for answer prediction. GeReA unlocks the use of MLLM as the implicit knowledge engine, surpassing all previous state-of-the-art methods on OK-VQA and A-OKVQA datasets, with test accuracies of 66.5% and 63.3% respectively.

Many thanks for your attention to our work!!!

If you find our project is helpful for your research, please kindly give us a :star2: and cite our paper :bookmark_tabs:   :)

## Citation

```
@article{ma2024gerea,
  title={GeReA: Question-Aware Prompt Captions for Knowledge-based Visual Question Answering},
  author={Ma, Ziyu and Li, Shutao and Sun, Bin and Cai, Jianfei and Long, Zuxiang and Ma, Fuyan},
  journal={arXiv preprint arXiv:2402.02503},
  year={2024}
}
```

## Getting Started

### Installation
To establish the environment, just run this code in the shell:
```
git clone https://github.com/Upper9527/GeReA.git
cd GeReA
conda env create -f requirements.yaml
conda activate gerea
```
That will create the environment ```gerea``` we used.

## Experimental Results

### Comparison with previous methods

![comparison](https://github.com/Upper9527/GeReA/blob/main/figs/1.png)

### Example visualization

![visualization](https://github.com/Upper9527/GeReA/blob/main/figs/2.png)

## Contact
If I cannot timely respond to your questions, you can send the email to maziyu@hnu.edu.cn.

## Acknowledgements
Our code is built on [FiD](https://github.com/facebookresearch/FiD) which is under the [LICENSE](https://github.com/facebookresearch/FiD/blob/main/LICENSE).
