import numpy as np
import torch
import argparse
import os
import torch.nn as nn
import random

from torch import optim
from importlib import import_module
from torch.utils.data import DataLoader
from ddrdataset import DDR_dataset
from datetime import datetime
import cv2
from functions import progress_bar
from torchnet import meter
import torch.nn.functional as F
from sklearn.metrics import f1_score,roc_auc_score, accuracy_score, cohen_kappa_score, precision_score
from efficientnet.multi_model import EfficientNet
from models import vgg_lanet, mobilenetv3_lanet, densenet_lanet, inceptionv3_lanet, resnet_lanet

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', default='res50', help='model')
parser.add_argument('--visname', '-vis', default='kaggle', help='visname')
parser.add_argument('--batch-size', '-bs', default=32, type=int, help='batch-size')
parser.add_argument('--lr', '-lr', default=1e-3, type=float, help='lr')
parser.add_argument('--epochs', '-eps', default=100, type=int, help='epochs')
parser.add_argument('--n_classes', '-n-cls', default=25, type=int, help='n-classes')
parser.add_argument('--save-dir', '-save-dir', default='./checkpoints', type=str, help='save-dir')
parser.add_argument('--printloss', '-pl', default=20, type=int, help='print-loss')
parser.add_argument('--seed', '-seed', type=int, default=12138)
parser.add_argument('--resume', '-re', type=str, default=None)
parser.add_argument('--test', '-test', type=bool, default=False)
parser.add_argument('--adaloss', '-adaloss', type=bool, default=False)


val_epoch = 1
test_epoch = 5

def parse_args():
    global args
    args = parser.parse_args()


def get_lr(cur, epochs):
    if cur < int(epochs * 0.3):
        lr = args.lr
    elif cur < int(epochs * 0.8):
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    return lr


def get_dynamic_lr(cur, epochs):
    power = 0.9
    lr = args.lr * (1 - cur / epochs) ** power
    return lr

best_acc = 0
best_kappa_clf = 0
best_kappa_grad = 0

best_test_acc = 0
best_test_kappa_clf = 0
best_test_kappa_grad = 0

def main():
    
    global best_acc
    global save_dir
    
    parse_args()

    #Set random seed for Pytorch and Numpy for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    if args.model == 'res50':
        net = resnet_lanet.resnet50(pretrained=True, adaloss=args.adaloss)
    elif args.model == 'res18':
        net = resnet_lanet.resnet18(pretrained=True, adaloss=args.adaloss)
    elif args.model == 'effb3':
        net = EfficientNet.from_pretrained('efficientnet-b3', num_classes=5, image_size=(512,512))
    elif args.model == 'dense121':
        net = densenet_lanet.densenet121(pretrained=True, adaloss=args.adaloss)
        net.classifier = nn.Linear(1024, 5)
    elif args.model == 'vgg':
        net = vgg_lanet.vgg16_bn(pretrained=True, adaloss=args.adaloss)
        net.classifier[6] = nn.Linear(4096, 5)
    elif args.model == 'mobilev3':
        net = mobilenetv3_lanet.mobilenet_v3_large(pretrained=True,adaloss=args.adaloss)
        net.classifier[3] = nn.Linear(1280, 5)
    elif args.model == 'inceptionv3':
        net = inceptionv3_lanet.inception_v3(pretrained=True,aux_logits=False,adaloss=args.adaloss)
        net.fc = nn.Linear(2048, 5)


    if args.adaloss:
        s1 = net.sigma1
        s2 = net.sigma2
    else:
        s1 = torch.zeros(1)
        s2 = torch.zeros(1)

    print(net)
    # exit()

    net = nn.DataParallel(net)
    net = net.cuda()

    dataset = DDR_dataset(train=True, val=False, test=False, multi=args.n_classes)
    valset = DDR_dataset(train=False, val=True, test=False, multi=args.n_classes)
    testset = DDR_dataset(train=False, val=False, test=True, multi=args.n_classes)

    dataloader = DataLoader(dataset,  shuffle=True, batch_size=args.batch_size, num_workers=8,pin_memory=True)
    valloader = DataLoader(valset, shuffle=False, batch_size=args.batch_size,num_workers=8,pin_memory=True)
    testloader = DataLoader(testset, shuffle=False, batch_size=args.batch_size,num_workers=8,pin_memory=True)

    # optim scheduler & crit
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    criterion_clf = nn.CrossEntropyLoss()
    criterion_clf = criterion_clf.cuda()
    criterion_grad = nn.CrossEntropyLoss()
    criterion_grad = criterion_grad.cuda()

    con_matx_clf = meter.ConfusionMeter(2)
    con_matx_grad = meter.ConfusionMeter(5)

    save_dir = './checkpoints/' + args.visname + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_log=open('./logs/'+args.visname+'.txt','a')   


    if args.test:
        if args.model == 'res50':
            # weight_dir = 'checkpoints/ddr512_res50_camcat_bs32/95.pkl'
            weight_dir = 'checkpoints/ddr512_res50_camcat_adl_bs32/179.pkl'
        elif args.model == 'res18':
            weight_dir = 'checkpoints/ddr512_res18_camcat_adl_bs32/171.pkl'
        elif args.model == 'effb3':
            # weight_dir = 'checkpoints/ddr512_effb3_camcat_bs32/76.pkl'
            weight_dir = 'checkpoints/ddr512_effb3_camcat_adl_bs32/172.pkl'
        elif args.model == 'vgg':
            # weight_dir = 'checkpoints/ddr512_vgg16_camcat_bs32/65.pkl'
            # weight_dir = 'checkpoints/ddr512_vgg16_camcat_bs32/73.pkl'
            # weight_dir = 'checkpoints/ddr512_vgg16_camcat_bs32/61.pkl'
            weight_dir = 'checkpoints/ddr512_vgg16_camcat_adl_bs32/65.pkl'
            # weight_dir = 'checkpoints/ddr512_vgg19_camcat_adl_bs32/88.pkl'


        elif args.model == 'dense121':
            # weight_dir = 'checkpoints/ddr512_dense121bn_camcat_bs32/73.pkl'
            weight_dir = 'checkpoints/ddr512_dense121_camcat_adl_bs32/199.pkl'
        elif args.model == 'mobilev3':
            # weight_dir = 'checkpoints/ddr512_mobilev3_camcat_bs32/82.pkl'
            weight_dir = 'checkpoints/ddr512_mobilev3_nodrop_camcat_bs32/50.pkl'
            # weight_dir = 'checkpoints/ddr512_mobilev3_camcat_adl_bs32/177.pkl'
            weight_dir = 'checkpoints/ddr512_mobilev3_nodrop_camcat_adl_bs32/163.pkl'
        elif args.model == 'inceptionv3':
            # weight_dir = 'checkpoints/ddr512_inceptionv3_camcat_bs32/40.pkl'
            # weight_dir = 'checkpoints/ddr512_inceptionv3_camcat_adl_bs32/173.pkl'
            # weight_dir = 'checkpoints/ddr512_inceptionv3_noaux_camcat_bs32/35.pkl'
            weight_dir = 'checkpoints/ddr512_inceptionv3_noaux_camcat_adl_bs32/267.pkl'

        epoch = int(weight_dir.split('/')[-1].split('.')[0])
        checkpoint = torch.load(weight_dir)
        state_dict = checkpoint['net']
        net.load_state_dict(state_dict, strict=True) 
        test_log=open('./logs/test.txt','a')   
        
        ddr_test(net, testloader, optimizer, epoch, test_log)
        exit()

    # resume from one epoch
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume,map_location='cuda:0')
            start_epoch = checkpoint['epoch']+1
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Model loaded from {}'.format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        start_epoch = 0



    for epoch in range(start_epoch, args.epochs):
        con_matx_clf.reset()
        con_matx_grad.reset()
        net.train()
        total_loss_clf = .0
        total_loss_grad = .0
        total = .0
        correct_clf = .0
        correct_grad = .0
        count = .0

        lr = get_dynamic_lr(epoch, args.epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for i, (x, label_clf, label_grad) in enumerate(dataloader):
            x = x.float().cuda()
            label_clf = label_clf.cuda()
            label_grad = label_grad.cuda()

            y_pred_clf, y_pred_grad = net(x)
            con_matx_clf.add(y_pred_clf.detach(),label_clf.detach())
            con_matx_grad.add(y_pred_grad.detach(),label_grad.detach())

            prediction_clf = y_pred_clf.max(1)[1]
            prediction_grad = y_pred_grad.max(1)[1]
            
            loss_clf = criterion_clf(y_pred_clf, label_clf)
            loss_grad = criterion_grad(y_pred_grad, label_grad)

            if args.adaloss:
                loss = torch.exp(-s1)*loss_clf+s1+torch.exp(-s2)*loss_grad+s2
            else:
                loss = loss_clf + loss_grad 

            total_loss_clf += loss_clf.item()
            total_loss_grad += loss_grad.item()
            total += x.size(0)
            correct_clf += prediction_clf.eq(label_clf).sum().item()
            correct_grad += prediction_grad.eq(label_grad).sum().item()
            count += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress_bar(i, len(dataloader), 'Loss clf: %.3f | Loss grad: %.3f | Acc clf: %.3f | Acc grad: %.3f'
                         % (total_loss_clf / (i + 1), total_loss_grad / (i + 1), 100. * correct_clf / total, 100. * correct_grad / total))

        
        if (epoch+1)%val_epoch == 0:
            ddr_val(net, valloader, optimizer, epoch, test_log, s1, s2)
               

@torch.no_grad()
def ddr_val(net, valloader, optimizer, epoch, test_log, s1, s2):
    global best_acc
    global best_kappa_clf
    global best_kappa_grad

    net = net.eval()
    total_acc = .0
    total_loss = .0
    correct_clf = .0
    correct_grad = .0
    total = .0
    count = .0
    con_matx_clf = meter.ConfusionMeter(2)
    con_matx_grad = meter.ConfusionMeter(5)


    pred_clf_list = []
    label_clf_list = []

    pred_grad_list = []
    label_grad_list = []

    for i, (x, label_clf, label_grad) in enumerate(valloader):
        x = x.float().cuda()
        label_clf = label_clf.cuda()
        label_grad = label_grad.cuda()

        y_pred_clf, y_pred_grad = net(x)
        con_matx_clf.add(y_pred_clf.detach(),label_clf.detach())
        con_matx_grad.add(y_pred_grad.detach(),label_grad.detach())      

        _, predicted_clf = y_pred_clf.max(1)
        _, predicted_grad = y_pred_grad.max(1)

        pred_clf_list.extend(predicted_clf.cpu().detach().tolist())
        label_clf_list.extend(label_clf.cpu().detach().tolist())

        pred_grad_list.extend(predicted_grad.cpu().detach().tolist())
        label_grad_list.extend(label_grad.cpu().detach().tolist())            

        total += x.size(0)
        count += 1
        correct_clf += predicted_clf.eq(label_clf).sum().item()
        correct_grad += predicted_grad.eq(label_grad).sum().item()

        progress_bar(i, len(valloader), ' Acc clf: %.3f|  Acc grad: %.3f'
                     % (100. * correct_clf / total, 100. * correct_grad / total))

    acc_clf = 100.0*accuracy_score(np.array(label_clf_list), np.array(pred_clf_list))
    kappa_clf = 100.0*cohen_kappa_score(np.array(label_clf_list), np.array(pred_clf_list), weights='quadratic')

    acc_grad = 100.0*accuracy_score(np.array(label_grad_list), np.array(pred_grad_list))
    kappa_grad = 100.0*cohen_kappa_score(np.array(label_grad_list), np.array(pred_grad_list), weights='quadratic')


    print('val epoch:', epoch, ' val acc clf: ', acc_clf, 'kappa clf: ', kappa_clf,
           'val acc grad: ', acc_grad, 'kappa grad: ', kappa_grad)

    test_log.write('Epoch:%d   Acc_clf:%.2f   kappa_clf:%.2f  Acc_grad:%.2f   kappa_grad:%.2f  s1:%.4f s2:%.4f \n'%(epoch,acc_clf, kappa_clf, acc_grad, kappa_grad, torch.exp(-s1).item(), torch.exp(-s2).item()))
    test_log.flush()  
    if kappa_grad >= best_kappa_grad:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
        }
        save_name = os.path.join(save_dir, str(epoch) + '.pkl')
        torch.save(state, save_name)
    if kappa_clf > best_kappa_clf:
        best_kappa_clf = kappa_clf
    if kappa_grad >= best_kappa_grad:
        best_kappa_grad = kappa_grad  


@torch.no_grad()
def ddr_test(net, testloader, optimizer, epoch, test_log):
    net = net.eval()
    total_acc = .0
    total_loss = .0
    correct_clf = .0
    correct_grad = .0
    total = .0
    count = .0
    con_matx_clf = meter.ConfusionMeter(2)
    con_matx_grad = meter.ConfusionMeter(5)


    pred_clf_list = []
    label_clf_list = []

    pred_grad_list = []
    label_grad_list = []

    for i, (x, label_clf, label_grad) in enumerate(testloader):
        x = x.float().cuda()
        label_clf = label_clf.cuda()
        label_grad = label_grad.cuda()

        y_pred_clf, y_pred_grad = net(x)
        con_matx_clf.add(y_pred_clf.detach(),label_clf.detach())
        con_matx_grad.add(y_pred_grad.detach(),label_grad.detach())      

        _, predicted_clf = y_pred_clf.max(1)
        _, predicted_grad = y_pred_grad.max(1)

        pred_clf_list.extend(predicted_clf.cpu().detach().tolist())
        label_clf_list.extend(label_clf.cpu().detach().tolist())

        pred_grad_list.extend(predicted_grad.cpu().detach().tolist())
        label_grad_list.extend(label_grad.cpu().detach().tolist())            

        total += x.size(0)
        count += 1
        correct_clf += predicted_clf.eq(label_clf).sum().item()
        correct_grad += predicted_grad.eq(label_grad).sum().item()

        progress_bar(i, len(testloader), ' Acc clf: %.3f|  Acc grad: %.3f'
                     % (100. * correct_clf / total, 100. * correct_grad / total))

    acc_clf = 100.0*accuracy_score(np.array(label_clf_list), np.array(pred_clf_list))
    kappa_clf = 100.0*cohen_kappa_score(np.array(label_clf_list), np.array(pred_clf_list), weights='quadratic')

    acc_grad = 100.0*accuracy_score(np.array(label_grad_list), np.array(pred_grad_list))
    kappa_grad = 100.0*cohen_kappa_score(np.array(label_grad_list), np.array(pred_grad_list), weights='quadratic')

    precision = 100.0*precision_score(np.array(label_grad_list), np.array(pred_grad_list),average='micro')
    f1 = 100.0*f1_score(np.array(label_grad_list), np.array(pred_grad_list),average='micro')
    print('test epoch:%d   acc clf:%.2f  kappa clf:%.2f  acc grad:%.2f  kappa grad:%.2f '% (epoch,acc_clf,kappa_clf,acc_grad,kappa_grad))
    print('precision:', precision)
    print('f1:', f1)
    print(con_matx_grad.value())

if __name__ == '__main__':
    main()