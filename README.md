# 【NN'24】Open-Pose 3D Zero-Shot Learning: Benchmark and Challenges

- [x] Release environment setting
- [x] Release open-pose benchmark datasets McGill<sup>‡</sup> 
- [x] Release datasets ModelNet40<sup>‡</sup>, ModelNet10<sup>‡</sup>, 
- [x] Release our baseline eval code CLIP-Based 
- [x] Release our baseline eval code Diffusion-Based

## Open-Pose Benchmark 
| Datasets | Total Classes | Seen/Unseen Classes | Train/Valid/Test Samples| Download |
|:---:|:---:|:---:|:---:|:---:|
|ModelNet40<sup>‡</sup>| 40| 30/-| 5852/1560/-| [google driver](https://drive.google.com/drive/folders/1OWvstxxpmXylxTSeGL7JqJGP3QjKp7p0?usp=drive_link) |
|ModelNet10<sup>‡</sup>| 10| -/10| -/-/908| [google driver](https://drive.google.com/drive/folders/1OWvstxxpmXylxTSeGL7JqJGP3QjKp7p0?usp=drive_link) |
|McGill<sup>‡</sup>| 19| -/14| -/-/115| [google driver](https://drive.google.com/drive/folders/1OWvstxxpmXylxTSeGL7JqJGP3QjKp7p0?usp=drive_link) |



<!--![avatar](doc/vis_benchmark.png)-->

## Our Baseline Method
![avatar](doc/overview.png)

### Environment
Our baseline (Diffusion-based or CLIP-based) could be conducted on one single RTX3090 or RTX4090. 
```    
conda env create -f op3dzsl.yaml
conda activtae op3dzsl
pip install git+https://github.com/openai/CLIP.git
```

Download the Diffusion pretrained model [google driver](https://drive.google.com/drive/folders/1OWvstxxpmXylxTSeGL7JqJGP3QjKp7p0?usp=drive_link) or [official website](https://huggingface.co/runwayml/stable-diffusion-v1-5). Rename the pretrained model as "model.ckpt" and put it in the directory "models/ldm/stable-diffusion-v1/".


### Baseline Evaluation

For our CLIP-Based baseline

```    
python baseline_eval/clip_eval.py
```

For our Diffusion-Based baseline

```    
python baseline_eval/diffusion_eval.py
```

## Citation
If you find this work useful in your research, please cite:
```
@article{zhao2024open,
  title={Open-Pose 3D zero-shot learning: Benchmark and challenges},
  author={Zhao, Weiguang and Yang, Guanyu and Zhang, Rui and Jiang, Chenru and Yang, Chaolong and Yan, Yuyao and Hussain, Amir and Huang, Kaizhu},
  journal={Neural Networks},
  pages={106775},
  year={2024}
}
```


If you utilize our open-pose datasets, it is necessary to cite the previous works from which they were developed: ModelNet40 and McGill.

```
@inproceedings{ModelNet,
  title={3d shapenets: A deep representation for volumetric shapes},
  author={Wu, Zhirong and Song, Shuran and Khosla, Aditya and Yu, Fisher and Zhang, Linguang and Tang, Xiaoou and Xiao, Jianxiong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1912--1920},
  year={2015}
}
```

```
@article{McGill,
  title={Retrieving articulated 3-D models using medial surfaces},
  author={Siddiqi, Kaleem and Zhang, Juan and Macrini, Diego and Shokoufandeh, Ali and Bouix, Sylvain and Dickinson, Sven},
  journal={Machine Vision and Application},
  volume={19},
  pages={261--275},
  year={2008}
}
```


## Acknowlegement
This project is not possible without multiple great opensourced codebases. We list some notable examples: 
[TZSL](https://github.com/ali-chr/Transductive_ZSL_3D_Point_Cloud), [PointCLIP](https://github.com/ZrrSkywalker/PointCLIP), [PointCLIPv2](https://github.com/yangyangyang127/PointCLIP_V2), [ReConCLIP](https://github.com/qizekun/ReCon), 
[CLIP2Point](https://github.com/tyhuang0428/CLIP2Point), [ULIP](https://github.com/salesforce/ULIP), 
[DiffCLIP](https://github.com/SitianShen/DiffCLIP), [Stable-Diffusion](https://github.com/runwayml/stable-diffusion) etc.