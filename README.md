# <img src="assets/images/logo.png" style="vertical-align: -10px;" :height="50px" width="50px"> VideoRoPE: What Makes for Good Video Rotary Position Embedding?


🚀🚀🚀 Official implementation of **VideoRoPE: What Makes for Good Video Rotary Position Embedding?**

<p align="center">
  <img src="assets/images/compare_table.png">
</p>

- **Authors**: [Xilin Wei*](https://github.com/Wiselnn570), [Xiaoran Liu*](https://scholar.google.de/citations?user=Qe6F4J4AAAAJ&hl=en), [Yuhang Zang](https://yuhangzang.github.io), [Xiaoyi Dong](https://lightdxy.github.io), [Pan Zhang](https://panzhang0212.github.io/), [Yuhang Cao](https://scholar.google.com/citations?user=sJkqsqkAAAAJ&hl=en), [Jian Tong](), [Haodong Duan](https://kennymckormick.github.io/), [Qipeng Guo](https://scholar.google.com/citations?user=k3mPGKgAAAAJ&hl=en), [Jiaqi Wang](https://myownskyw7.github.io/), [Xipeng Qiu](https://xpqiu.github.io/en.html), [Dahua Lin](http://dahua.site/)
- **Institutes**: Fudan University; Shanghai AI Laboratory; Shanghai Innovation Institute
- **Resources**: [📖[Paper](https://arxiv.org/pdf/2502.05173)] [[🏠Project Page](https://wiselnn570.github.io/VideoRoPE/)] [[🤗Huggingface]()]
## 💡 Highlights

- 🔥 **Four Key Positional Encoding Schemes:** We present an analysis of four key properties essential for RoPE when applied to video. Motivated by this analysis, we propose **VideoRoPE** including **Low-frequency Temporal Allocation (LTA)**, **Diagonal Layout (DL)**, and **Adjustable Temporal Spacing (ATS)** to satisfy all four properties.
- 🔥 **A Challenging Video Haystack Retrieval Benchmark:** We introduce the challenging **V-NIAH-D** task to expose the drawbacks of current position embedding designs regarding frequency allocation. Our findings reveal that existing Video LLMs are easily misled to frequency-based distractors.
- 🔥 **Excellent Performance:** Extensive experiments demonstrate that VideoRoPE consistently achieves superior performance compared to other RoPE variants. For example, VideoRoPE outperforms previous M-RoPE on long video retrieval (+12.4 on V-NIAH, +12.4 on V-NIAH-D), video understanding (+2.9 on LongVideoBench, +4.5 on MLVU, +1.7 on Video-MME) and hallucination (+11.9 on VideoHallucer) benchmarks.

<!-- ## 📜 News

**[2024/10/1]** ShareGPT4Video was accepted by NeurIPS 2024 D&B track!

**[2024/7/1]** The code about batch-inference of ShareCaptioner-Video is available now!

**[2024/6/11]** The web demo and local demo of ShareCaptioner-Video are available now!

**[2024/6/11]** The web demo and local demo of ShareGPT4Video-8B are available now!

**[2024/6/7]** Our paper has been featured as [HuggingFace Daily Papers](https://huggingface.co/papers?date=2024-06-07) and ranked 1st in 6.7.

**[2024/5/27]** The [ShareGPT4Video-8B](https://huggingface.co/Lin-Chen/sharegpt4video-8b) model is released!

**[2024/5/26]** The [ShareGPT4Video dataset](https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video) and [project page](https://sharegpt4video.github.io/) are released! -->

## 👨‍💻 Todo

- [] VideoRoPE Implementation with *transformers*
- [] VideoRoPE Implementation with *vLLM*
- [] V-NIAH-D Release
- [] VideoRoPE-Based Model Checkpoints

<!-- ## Quick Usage

You can directly use our ShareGPT4Video model for conversation with your own video by the following command:

```
python run.py --model-path Lin-Chen/sharegpt4video-8b --video examples/yoga.mp4 --query Describe this video in detail.
```

Or you can build your local demo to enjoy our ShareGPT4Video-8B with the following command:

```
python app.py
```

You can build your local demo for enjoying our ShareCaptioner-Video with the following command:

```
cd captioner

python app.py
```

## Install

```bash
git clone https://github.com/ShareGPT4Omni/ShareGPT4Video
conda create -n share4video python=3.10 -y
conda activate share4video

cd ShareGPT4Video
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Train

To validate the effectiveness of high-quality video captions for helping to improve the LVLMs' comprehension capabilities. We choose the [VideoLLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA?tab=readme-ov-file) and [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID) models as our baselines. The SFT data used for both models is LLaVA-mix665K image data plus VideoChatGPT-100K video data. We replace 28K caption data in VideoChatGPT-100K with 28K high quality caption data from ShareGPT4Video. Next, we take VideoLLaVA as the example.

You need to follow the [instructions](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/TRAIN_AND_VALIDATE.md) in VideoLLaVA to prepare the images and videos first, then download the 28K videos used in ShareGPT4Video from [HuggingFace](https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video/tree/main/zip_folder) (only involves bdd100k, ego4d, and panda).

Finally, you can specify the llava_v1_5_mix665k_with_video_chatgpt72k_share4video28k.json file in the [finetune.sh](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/scripts/v1_5/finetune.sh) to perform the SFT to reproduce the results in the paper. -->

<!-- ## ✒️ Citation

If you find our work helpful for your research, please consider giving a star ⭐ and citation 📝

```bibtex

@article{chen2024sharegpt4video,
  title={ShareGPT4Video: Improving Video Understanding and Generation with Better Captions},
  author={Chen, Lin and Wei, Xilin and Li, Jinsong and Dong, Xiaoyi and Zhang, Pan and Zang, Yuhang and Chen, Zehui and Duan, Haodong and Lin, Bin and Tang, Zhenyu and others},
  journal={arXiv preprint arXiv:2406.04325},
  year={2024}
}

@article{chen2023sharegpt4v,
  title={ShareGPT4V: Improving Large Multi-Modal Models with Better Captions},
  author={Chen, Lin and Li, Jisong and Dong, Xiaoyi and Zhang, Pan and He, Conghui and Wang, Jiaqi and Zhao, Feng and Lin, Dahua},
  journal={arXiv preprint arXiv:2311.12793},
  year={2023}
}

@article{chen2024we,
  title={Are We on the Right Way for Evaluating Large Vision-Language Models?},
  author={Chen, Lin and Li, Jinsong and Dong, Xiaoyi and Zhang, Pan and Zang, Yuhang and Chen, Zehui and Duan, Haodong and Wang, Jiaqi and Qiao, Yu and Lin, Dahua and others},
  journal={arXiv preprint arXiv:2403.20330},
  year={2024}
}
``` -->

## ❤️ Acknowledgments

- [transformers](https://github.com/huggingface/transformers): the codebase we built upon. Thanks for their wonderful work.
- [vLLM](https://github.com/PKU-YuanGroup/Open-Sora-Plan): an excellent open-source codebase for high-throughput and memory-efficient inference. Thanks for their wonderful work.
