---
license: apache-2.0
---


<br>

# Vim Model Card

## Model Details

Vision Mamba (Vim) is a generic backbone trained on the ImageNet-1K dataset for vision tasks.

- **Developed by:** [HUST](https://english.hust.edu.cn/), [Horizon Robotics](https://en.horizon.cc/), [BAAI](https://www.baai.ac.cn/english.html)
- **Model type:** A generic vision backbone based on the bidirectional state space model (SSM) architecture.
- **License:** Non-commercial license


### Model Sources

- **Repository:** https://github.com/hustvl/Vim
- **Paper:** https://arxiv.org/abs/2401.09417

## Uses

The primary use of Vim is research on vision tasks, e.g., classification, segmentation, detection, and instance segmentation, with an SSM-based backbone.
The primary intended users of the model are researchers and hobbyists in computer vision, machine learning, and artificial intelligence.

## How to Get Started with the Model

- You can replace the backbone for vision tasks with the proposed Vim: https://github.com/hustvl/Vim/blob/main/vim/models_mamba.py
- Then you can load this checkpoint and start training.

## Training Details

Vim is pretrained on ImageNet-1K with classification supervision.
The training data is around 1.3M images from [ImageNet-1K dataset](https://www.image-net.org/challenges/LSVRC/2012/).
See more details in this [paper](https://arxiv.org/abs/2401.09417).

## Evaluation

Vim-base is evaluated on ImageNet-1K val set, and achieves 81.9% Top-1 Acc. See more details in this [paper](https://arxiv.org/abs/2401.09417).

## Additional Information

### Citation Information

```
 @article{vim,
  title={Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model},
  author={Lianghui Zhu and Bencheng Liao and Qian Zhang and Xinlong Wang and Wenyu Liu and Xinggang Wang},
  journal={arXiv preprint arXiv:2401.09417},
  year={2024}
}
```

