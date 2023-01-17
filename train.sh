# ddr
# res50
# CUDA_VISIBLE_DEVICES=0,1 python main_base.py --dataset ddr --visname ddr512_res50_base2_bs32 -bs 32 --lr 1e-3 --n_classes 2
# CUDA_VISIBLE_DEVICES=0,1 python main_base.py --dataset ddr --visname ddr512_res50_base5_bs32 -bs 32 --lr 1e-3 --n_classes 5
# CUDA_VISIBLE_DEVICES=0,1 python main_lanet.py --dataset ddr --visname ddr512_res50_cam_bs32 -bs 32
# CUDA_VISIBLE_DEVICES=0,1 python main_lanet.py --dataset ddr --visname ddr512_res50_camcat_adl_bs32 -bs 32 --adaloss True --epoch 200

# effb3
# CUDA_VISIBLE_DEVICES=0,1 python main_base.py --dataset ddr --model effb3 --visname ddr512_effb3_base2_bs32 -bs 32 --n_classes 2
# CUDA_VISIBLE_DEVICES=0,1 python main_base.py --dataset ddr --model effb3 --visname ddr512_effb3_base5_bs32 -bs 32 --n_classes 5
# CUDA_VISIBLE_DEVICES=0,1 python main_lanet.py --dataset ddr --model effb3 --visname ddr512_effb3_camcat_bs32 -bs 32
# CUDA_VISIBLE_DEVICES=0,1 python main_lanet.py --dataset ddr --model effb3 --visname ddr512_effb3_camcat_adl_bs32 -bs 32 --adaloss True --epoch 200

# vgg16_bn
# CUDA_VISIBLE_DEVICES=0,1 python main_base.py --dataset ddr --model vgg --visname ddr512_vgg16_base5_bs32 -bs 32
# CUDA_VISIBLE_DEVICES=0,1 python main_base.py --dataset ddr --model vgg --visname ddr512_vgg16_base2_bs32 -bs 32 --n_classes 2
# CUDA_VISIBLE_DEVICES=0,1 python main_lanet.py --dataset ddr --model vgg --visname ddr512_vgg16_camcat_bs32 -bs 32
# CUDA_VISIBLE_DEVICES=0,1 python main_lanet.py --dataset ddr --model vgg --visname ddr512_vgg16_camcat_adl_bs32 -bs 32 --adaloss True


# mobilnetv3
# CUDA_VISIBLE_DEVICES=0,1 python main_base.py --dataset ddr --model mobilev3 --visname ddr512_mobilev3_base5_bs32 -bs 32
# CUDA_VISIBLE_DEVICES=0,1 python main_base.py --dataset ddr --model mobilev3 --visname ddr512_mobilev3_base2_bs32 -bs 32 --n_classes 2
# CUDA_VISIBLE_DEVICES=0,1 python main_lanet.py --dataset ddr --model mobilev3 --visname ddr512_mobilev3_nodrop_camcat_bs32 -bs 32
# CUDA_VISIBLE_DEVICES=0,1 python main_lanet.py --dataset ddr --model mobilev3 --visname ddr512_mobilev3_nodrop_camcat_adl_bs32 -bs 32 --adaloss True --epoch 200

# inception_v3
# CUDA_VISIBLE_DEVICES=0,1 python main_base.py --dataset ddr --model inceptionv3 --visname ddr512_inceptionv3_noaux_base5_bs32 -bs 32
# CUDA_VISIBLE_DEVICES=0,1 python main_base.py --dataset ddr --model inceptionv3 --visname ddr512_inceptionv3_noaux_base2_bs32 -bs 32 --n_classes 2
# CUDA_VISIBLE_DEVICES=0,1 python main_lanet.py --dataset ddr --model inceptionv3 --visname ddr512_inceptionv3_noaux_camcat_bs32 -bs 32
# CUDA_VISIBLE_DEVICES=0,1 python main_lanet.py --dataset ddr --model inceptionv3 --visname ddr512_inceptionv3_noaux_camcat_adl_bs32 -bs 32 --adaloss True --epoch 300


# densenet
# CUDA_VISIBLE_DEVICES=0,1 python main_base.py --dataset ddr --model dense121 --visname ddr512_dense121_base5_bs32 -bs 32
# CUDA_VISIBLE_DEVICES=0,1 python main_base.py --dataset ddr --model dense121 --visname ddr512_dense121_base2_bs32 -bs 32 --n_classes 2
# CUDA_VISIBLE_DEVICES=0,1 python main_lanet.py --dataset ddr --model dense121 --visname ddr512_dense121bn_camcat_bs32 -bs 32
# CUDA_VISIBLE_DEVICES=0,1 python main_lanet.py --dataset ddr --model dense121 --visname ddr512_dense121_camcat_adl_bs32 -bs 32 --adaloss True --epoch 200


# test
# CUDA_VISIBLE_DEVICES=0,1 python main_base.py --dataset ddr --model inceptionv3 --visname tests -bs 32 --n_classes 5 --test True
# CUDA_VISIBLE_DEVICES=0,1 python main_lanet.py --dataset ddr --model vgg --visname tests -bs 32 --test True --adaloss True

