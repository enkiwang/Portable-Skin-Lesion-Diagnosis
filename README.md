# Portable-Skin-Lesion-Diagnosis
This repository provides our codes to train a portable skin lesion diagnosis model based on our proposed knowledge distillation framework. The portable student model is trained by distilling, integrating, and transferring diverse knowledge (D-KD) extracted from a pretrained teacher model. A self-supervised variant (SSD-KD) is also employed to guide the student to capture richer informative knowledge for skin lesion diagnosis. 

## Environment configuration
Our KD implementations were tested under a linux-centos7-x86\_64 system with GPU cards as NVIDIA Tesla V100 (16GB/32GB memory). Please refer to python dependencies in the `requirements.txt` file. You can install these required dependencies via,
```python
pip3 install -r requirements.txt
```


## Dataset preparation
Please download ISIC 2019, a large-scale dermoscopic image dataset, from [this link](https://challenge2019.isic-archive.com/). After downloading ISIC 2019, please put it in the data/ folder, then preprocess this dataset using the provided script,
 ```bash
 bash ./run_preprocess.sh
 ```

## Train a portable diagnosis model
Please go to the experiment/isic/ directory, download the teacher model ResNet-50 pretrained by us in [Google Drive](https://drive.google.com/file/d/1yz0nh3811KoyVqz_ln_JsLdC--ssAQD6/view?usp=sharing), and put it in the results/ folder. You can also train it from scratch using `isic.py` which was borrowed from the [MetaBlock](https://github.com/paaatcha/MetaBlock/tree/main/benchmarks/isic) project. 

To train the portable student MobileNet-V2 model, you can choose to use D-KD or SSD-KD:

* For the D-KD method, please perform training following a demo,
```python
python3 isic_d_kd.py with '_kd_method="d_kd"' '_lambd_drkd=1' '_lambd_crkd=1000'
```
You can also download our trained student model by D-KD from [Google Drive](https://drive.google.com/file/d/1b5Wl3lQM5qmsQvftQjy7aQWN_y5Ha8er/view?usp=sharing).

* For the SSD-KD method, please perform training following a demo,
```python
python3 isic_ssd_kd.py with '_kd_method="ssd_kd"' '_lambd_drkd=1' '_lambd_crkd=1000'
```
You can also download our trained student model by SSD-KD from [Google Drive](https://drive.google.com/file/d/1UkRX0c2moP906tqcaT9hCiS9WaoWJdld/view?usp=sharing).

If you encounter possible issues regarding this code, please do not hesitate to [contact me](mailto:yongweiw@ece.ubc.ca).


If you find our code useful in your research, please consider citing our work:

```bib
@article{wang2023ssd,
  title={Ssd-kd: A self-supervised diverse knowledge distillation method for lightweight skin lesion classification using dermoscopic images},
  author={Wang, Yongwei and Wang, Yuheng and Cai, Jiayue and Lee, Tim K and Miao, Chunyan and Wang, Z Jane},
  journal={Medical Image Analysis},
  volume={84},
  pages={102693},
  year={2023},
  publisher={Elsevier}
}
```


## Acknowledgements
Many thanks to the following open-sourced repositories that we borrowed which have greatly facilitated our implementations:

* [https://github.com/paaatcha/MetaBlock]

* [https://github.com/paaatcha/raug]

* [https://raw.githubusercontent.com/AberHu/Knowledge-Distillation-Zoo/]

* [https://github.com/xuguodong03/SSKD]






 

