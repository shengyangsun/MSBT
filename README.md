# MultiScale Bottleneck Transformer (MSBT)

**[ICME 2024] Multi-scale Bottleneck Transformer for Weakly Supervised Multimodal Violence Detection**

Shengyang Sun, Xiaojin Gong

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-scale-bottleneck-transformer-for-weakly/anomaly-detection-in-surveillance-videos-on-2)](https://paperswithcode.com/sota/anomaly-detection-in-surveillance-videos-on-2?p=multi-scale-bottleneck-transformer-for-weakly)
<a href='https://arxiv.org/abs/2405.05130'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>

## Framework
<p align="center">
    <img src=framework.png width="800" height="300"/>
</p>
An overview of the proposed framework. It includes three unimodal encoders, a multimodal fusion module, and a global encoder for multimodal feature generation. Each unimodal encoder consists of a modality-specific feature extraction backbone and a linear projection layer for tokenization and a modality-shared transformer for context aggregation within one modality. The fusion module contains a multi-scale bottleneck transformer (MSBT) to fuse any pair of modalities and a sub-module to weight concatenated fused features. 
	The global encoder, implemented by a transformer, aggregates context over all modalities. Finally, the produced multimodal features are fed into a regressor to predict anomaly scores.

## News
- [2024.05.09] ⭐️ Release the **Multi-scale Bottleneck Transformer (MSBT)**, we encourage you to integrate the MSBT module into your framework to enhance the performance of feature fusion, the detailed implementation of MSTB can be referred to *MultiScaleBottleneckTransformer.py*.

## Performance on XD-Violence Dataset
| Method | Modality |AP (%) |
| ----------| :------: | :----:|
| MSBT (Ours)| RGB & Audio | 82.54 |
| MSBT (Ours)| RGB & Flow | 80.68 |
| MSBT (Ours)| Audio & Flow | 77.47 |
| MSBT (Ours)| RGB & Audio & Flow | 84.32 |


## Requirements  

    python==3.7.13
    torch==1.11.0  
    cuda==11.3
    numpy==1.21.5

## Extracted Features of XD-Violence 

  The extracted features can be downloaded from this [official page](https://roc-ng.github.io/XD-Violence). We use the **RGB**, **Flow**, and **Audio** features in this paper, you should download the features from that page and arrange the features paths into path lists in the **list/** folder one by one as follows:

    I3D/RGB -> rgb.list
    I3D/RGBTest -> rgb_test.list
    I3D/Flow -> flow.list
    I3D/FlowTest -> flow_test.list
    vggish-features/Train -> audio.list
    vggish-features/Test -> audio_test.list

## Training

    python main.py 

## Testing

    python main.py --eval --model_path ckpt/MSBT_best_84.32.pkl 

## Citation
If you find our paper useful, hope you can star our repo and cite our paper as follows:
```
@inproceedings{sun2024multiscale,
  title={Multi-scale Bottleneck Transformer for Weakly Supervised Multimodal Violence Detection},
  author={Sun, Shengyang and Gong, Xiaojin},
  booktitle={2024 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
```
## License
This project is released under the MIT License.

## Acknowledgements
Some codes are based on [MACIL_SD](https://github.com/JustinYuu/MACIL_SD) and [XDVioDet](https://github.com/Roc-Ng/XDVioDet), we sincerely thank them for their contributions.
