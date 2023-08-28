# LANet

This repo covers the official implementation of paper DIABETIC RETINOPATHY GRADING WITH WEAKLY-SUPERVISED LESION PRIORS

## Dataset
Download the DDR Dataset from [DDR (github)](https://github.com/nkicsl/DDR-dataset)

## Train

We provide the implementation of LANet with different backbones, including resnet, densenet, vgg, mobilenet, efficientnet, inceptionv3. Take resnet as an example:

### baseline model
DR recognition (binary classification)
```
python main_base.py --n_classes 2 --model res50 --visname ddr_res50_base2
```

DR grading (5-grade classification)
```
python main_base.py --n_classes 5 --model res50 --visname ddr_res50_base5
```


### lanet model
LANet (w/o adaptive loss)
```
python main_lanet.py --model res50 --visname ddr_res50_lanet
```

LANet (w/ adaptive loss)
```
python main_lanet.py --model res50 --adaloss True --visname ddr_res50_lanet_adl
```

## Test
baseline model
```
python main_base.py --dataset ddr --model res50 --visname tests --n_classes 5 --test True
```

LANet (w/ adaptive loss)
```
python main_lanet.py --dataset ddr --model res50 --visname tests --adaloss True --test True 
```


## Reference 
```
@inproceedings{hou2023diabetic,
  title={Diabetic Retinopathy Grading with Weakly-Supervised Lesion Priors},
  author={Hou, Junlin and Xiao, Fan and Xu, Jilan and Feng, Rui and Zhang, Yuejie and Zou, Haidong and Lu, Lina and Xue, Wenwen},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

