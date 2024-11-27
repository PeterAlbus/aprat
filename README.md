# Adaptive Adapter Routing for Long-Tailed Class-Incremental Learning
<div align="center">

<div>
    <a href='http://www.lamda.nju.edu.cn/qizh' target='_blank'>Zhi-Hong Qi</a>&emsp;
    <a href='http://www.lamda.nju.edu.cn/zhoudw' target='_blank'>Da-Wei Zhou</a>&emsp;
    <a>Yiran Yao</a>&emsp;
    <a href='http://www.lamda.nju.edu.cn/yehj' target='_blank'>Han-Jia Ye</a>&emsp;
    <a href='http://www.lamda.nju.edu.cn/zhandc' target='_blank'>De-Chuan Zhan</a>
</div>
<div>
School of Artificial Intelligence, State Key Laboratory for Novel Software Technology, Nanjing University&emsp;

</div>
</div>



<div align="center">

</div>


The code repository for "[Adaptive Adapter Routing for Long-Tailed Class-Incremental Learning](https://arxiv.org/abs/2409.07446)" (MJL 2024) in PyTorch. 

<!-- 
 If you use any content of this repo for your work, please cite the following bib entry: 

    @article{zhou2024revisiting,
        author = {Zhou, Da-Wei and Cai, Zi-Wen and Ye, Han-Jia and Zhan, De-Chuan and Liu, Ziwei},
        title = {Revisiting Class-Incremental Learning with Pre-Trained Models: Generalizability and Adaptivity are All You Need},
        journal = {International Journal of Computer Vision},
        year = {2024}
    } -->




## 🔧 Requirements
###  Environment 
1. [torch 1.11.0](https://github.com/pytorch/pytorch)
2. [torchvision 0.12.0](https://github.com/pytorch/vision)
3. [timm 0.6.12](https://github.com/huggingface/pytorch-image-models)


### Dataset 
We provide the processed datasets as follows:
- **CIFAR100**: will be automatically downloaded by the code.

- **ImageNet-R**: Google Drive: [link](https://drive.google.com/file/d/1SG4TbiL8_DooekztyCVK8mPmfhMo8fkR/view?usp=sharing) or Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EU4jyLL29CtBsZkB6y-JSbgBzWF5YHhBAUz1Qw8qM2954A?e=hlWpNW)

- **ObjectNet**: Onedrive: [link](https://entuedu-my.sharepoint.com/:u:/g/personal/n2207876b_e_ntu_edu_sg/EZFv9uaaO1hBj7Y40KoCvYkBnuUZHnHnjMda6obiDpiIWw?e=4n8Kpy) You can also refer to the [filelist](https://drive.google.com/file/d/147Mta-HcENF6IhZ8dvPnZ93Romcie7T6/view?usp=sharing) and processing [code](https://github.com/zhoudw-zdw/RevisitingCIL/issues/2#issuecomment-2280462493) if the file is too large to download. 

These subsets are sampled from the original datasets. Please note that I do not have the right to distribute these datasets. If the distribution violates the license, I shall provide the filenames instead.

You need to modify the path of the datasets in `./utils/data.py`  according to your own path. 

## 💡 Running scripts

To prepare your JSON files, refer to the settings in the `exps` folder and run the following command. The results can be found in the `logs` folder.

```
python main.py --config ./exps/[configname].json
```


## 🎈 Acknowledgement
This repo is based on [LAMDA-PILOT](https://github.com/sun-hailong/LAMDA-PILOT), [CIL_Survey](https://github.com/zhoudw-zdw/CIL_Survey) and [PyCIL](https://github.com/G-U-N/PyCIL).


## 💭 Correspondence
If you have any questions, please contact me via [email](mailto:qizh@lamda.nju.edu.cn) or open an [issue](https://github.com/vita-qzh/APART/issues/new).
