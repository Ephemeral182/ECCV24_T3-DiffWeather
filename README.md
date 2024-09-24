# Teaching Tailored to Talent: Adverse Weather Restoration via Prompt Pool and Depth-Anything Constraint (ECCV'2024)


> *<a href="https://ephemeral182.github.io">Sixiang Chen</a><sup>1</sup>&nbsp;&nbsp;&nbsp;
<a href="https://owen718.github.io">Tian Ye</a><sup>1</sup>&nbsp;&nbsp;&nbsp;
Kai Zhang</a><sup>1</sup>&nbsp;&nbsp;&nbsp;
<a href="https://ge-xing.github.io/">Zhaohu Xing</a><sup>1</sup>&nbsp;&nbsp;&nbsp;
<a href="https://lyl1015.github.io/">Yunlong Lin<sup>2</sup>&nbsp;&nbsp;&nbsp;
<a href="https://sites.google.com/site/indexlzhu/home">Lei Zhu</a><sup>1,3 ✉️</sup>&nbsp;&nbsp;&nbsp;*

> *<sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou)&nbsp;&nbsp;&nbsp;
<sup>2</sup>Xiamen University&nbsp;&nbsp;&nbsp;<br>
<sup>3</sup>The Hong Kong University of Science and Technology&nbsp;&nbsp;&nbsp;*


<em>European Conference on Computer Vision <strong>(ECCV)</strong>, 2024, MiCo Milano</em>
</div>

</br>

[![paper](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://ephemeral182.github.io/UDR_S2Former_deraining/)
[![supplement](https://img.shields.io/badge/Supplementary-Material-B85252)](https://ephemeral182.github.io/UDR_S2Former_deraining/)
[![project](https://img.shields.io/badge/Project-Presentation-F9D371)](https://ephemeral182.github.io/UDR_S2Former_deraining/)


[<img src="images/ban1.jpg" height="320px"/>](https://imgsli.com/Mjk5ODA0) [<img src="images/ban2.jpg" height="320px"/>](https://imgsli.com/Mjk5ODEx)
[<img src="images/ban3.jpg" height="320px"/>](https://imgsli.com/Mjk5ODEy) [<img src="images/ban7.jpg" height="320px"/>](https://imgsli.com/Mjk5ODE0) 
  
## :fire: News

## Abstract

> *Recent advancements in adverse weather restoration have shown potential, yet the unpredictable and varied combinations of weather degradations in the real world pose significant challenges. Previous methods typically struggle with dynamically handling intricate degradation combinations and carrying on background reconstruction precisely, leading to performance and generalization limitations. Drawing inspiration from prompt learning and the "<u>T</u>eaching <u>T</u>ailored to <u>T</u>alent" concept, we introduce a novel pipeline, <i><strong>T<sup>3</sup>-DiffWeather</strong></i>. Specifically, we employ a prompt pool that allows the network to autonomously combine sub-prompts to construct weather-prompts, harnessing the necessary attributes to adaptively tackle unforeseen weather input. Moreover, from a scene modeling perspective, we incorporate general prompts constrained by Depth-Anything feature to provide the scene-specific condition for the diffusion process. Furthermore, by incorporating contrastive prompt loss, we ensures distinctive representations for both types of prompts by a mutual pushing strategy. Experimental results demonstrate that our method achieves state-of-the-art performance across various synthetic and real-world datasets, markedly outperforming existing diffusion techniques in terms of computational efficiency.*

## Installation
:satisfied: Our <i><strong>T<sup>3</sup>-DiffWeather</strong></i> is built in Pytorch=2.0.1, we train and test it on Ubuntu=20.04 environment (Python=3.8+, Cuda=11.6).

For installing, please follow these instructions:
```
conda create -n py38 python=3.8.16
conda activate py38
pip3 install torch torchvision torchaudio
pip3 install -r requirements.txt  
```

## Dataset
:open_file_folder: We train our model in the mixed adverse weather data and evaluate it in <strong>(Raindrop)</strong>, <strong>(Rainhaze (Test1))</strong> and <strong>(Snow100K)</strong>. The download links of datasets are provided.
<table>
  <tr>
    <th align="left">Adverse Weather</th>
    <th align="center">Adverse Weather</th>
    <th align="center">Raindrop</th>
    <th align="center">Test1</th>
    <th align="center">Snow100K</th>
  </tr>
  <tr>
    <td align="left">Link</td>
    <td align="center"><a href="https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html">Download</a></td>
    <td align="center"><a href="">Download</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1rk_ZdYuMIOTYeZxqUwi0YA">Download</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1L2Ral6sz2iIHMhHeHHsS">Download</a></td>
  </tr>
    <tr>
    <td align="left">Code</td>
    <td align="center"><a href="https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html">Download</a></td>
    <td align="center"><a href="https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html">Download</a></td>
    <td align="center">ifbm</td>
    <td align="center">6rtc</td>
  </tr>
 </table>

## Training Stage

:yum: Our training process is built upon pytorch_lightning, rather than the conventional torch framework. Please run the code below to begin training UDR-S<sup>2</sup>Former on various benchmarks (raindrop_syn,raindrop_real,agan,  
rain200h,rain200l). Example usage to training our model in raindrop_real:
```python
python train.py fit -c config/config_pretrain_raindrop_real.yaml
```
The logs and checkpoints are saved in ‘**tb_logs/udrs2former**‘.

 
## Citation 
```
@InProceedings{chen2024teaching,
    title     = {Teaching Tailored to Talent: Adverse Weather Restoration via Prompt Pool and Depth-Anything Constraint},
    author    = {Chen, Sixiang and Ye, Tian and Zhang, Kai and Xing, Zhaohu and Lin, Yunlong and Zhu, Lei}, 
    booktitle = {European conference on computer vision},
    year      = {2024},
    organization={Springer}
}
```
## Contact
If you have any questions, please contact the email ephemeral182@gmail.com or schen691@connect.hkust-gz.edu.cn(will be replaced soon) 
