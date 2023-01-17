import numpy as np
import torch
import argparse
import os
import torch.nn as nn
import cv2
import random
import torch.nn.functional as F

from torch import optim
from importlib import import_module
from torch.utils.data import DataLoader
from ddrdataset import DDR_dataset

from datetime import datetime
from functions import progress_bar
from torchnet import meter
from sklearn.metrics import f1_score,roc_auc_score, accuracy_score, cohen_kappa_score
from efficientnet.model import EfficientNet
from models import densenet, mobilenetv3, vgg, inceptionv3, resnet

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

parser = argparse.ArgumentParser()
parser.add_argument('--model', '-m', default='res50', help='model')
parser.add_argument('--visname', '-vis', default='kaggle', help='visname')
parser.add_argument('--batch-size', '-bs', default=32, type=int, help='batch-size')
parser.add_argument('--lr', '-lr', default=1e-3, type=float, help='lr')
parser.add_argument('--epochs', '-eps', default=100, type=int, help='epochs')
parser.add_argument('--n_classes', '-n-cls', default=5, type=int, help='n-classes')
parser.add_argument('--save-dir', '-save-dir', default='./checkpoints', type=str, help='save-dir')
parser.add_argument('--printloss', '-pl', default=20, type=int, help='print-loss')
parser.add_argument('--seed', '-seed', type=int, default=12138)
parser.add_argument('--resume', '-re', type=str, default=None)
parser.add_argument('--test', '-test', type=bool, default=False)


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
best_kappa_clf_fusion = 0

best_test_acc = 0
best_test_kappa_clf = 0


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
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if args.model == 'res50':
        net = resnet.resnet50(pretrained=True)
        net.fc = nn.Linear(2048, args.n_classes)
    elif args.model == 'effb3':
        net = EfficientNet.from_pretrained('efficientnet-b3', num_classes=args.n_classes)
    elif args.model == 'dense121':
        net = densenet.densenet121(pretrained=True)
        net.classifier = nn.Linear(1024, args.n_classes)
    elif args.model == 'vgg':
        net = vgg.vgg16_bn(pretrained=True)
        net.classifier[6] = nn.Linear(4096, args.n_classes)
    elif args.model == 'mobilev3':
        net = mobilenetv3.mobilenet_v3_large(pretrained=True)
        net.classifier[3] = nn.Linear(1280, args.n_classes)
    elif args.model == 'inceptionv3':
        net = inceptionv3.inception_v3(pretrained=True, aux_logits=False)
        net.fc = nn.Linear(2048, args.n_classes)
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
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5) #1e-5

    # loss
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    con_matx = meter.ConfusionMeter(args.n_classes)

    save_dir = './checkpoints/' + args.visname + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    test_log=open('./logs/'+args.visname+'.txt','a')   

    # resume from one epoch
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume,map_location='cuda')
            start_epoch = checkpoint['epoch']+1
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('Model loaded from {}'.format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        start_epoch = 0

    if args.test:
        if args.model == 'res50':
            # weight_dir ='checkpoints1009/ddr512_res50_single_bs32/23.pkl'
            weight_dir ='checkpoints1009/ddr512_res50_single2_bs32/40.pkl'
        elif args.model == 'effb3':
            # weight_dir = 'checkpoints1009/ddr512_effb3_single5_bs32/97.pkl'
            weight_dir = 'checkpoints1009/ddr512_effb3_single2_bs32/67.pkl'
        elif args.model == 'vgg':
            # weight_dir = 'checkpoints1009/ddr512_vgg16_single2_bs32/88.pkl'
            weight_dir = 'checkpoints1009/ddr512_vgg16_single5_bs32/48.pkl'
        elif args.model == 'dense121':
            # weight_dir = 'checkpoints1009/ddr512_dense121_single5_bs32/52.pkl'
            weight_dir = 'checkpoints1009/ddr512_dense121_single2_bs32/38.pkl'
        elif args.model == 'mobilev3':
            # weight_dir = 'checkpoints1009/ddr512_mobilev3_single2_bs32/41.pkl'
            weight_dir = 'checkpoints1009/ddr512_mobilev3_single5_bs32/65.pkl'
        elif args.model == 'inceptionv3':
            # weight_dir = 'checkpoints1009/ddr512_inceptionv3_single2_bs32/66.pkl'
            # weight_dir = 'checkpoints1009/ddr512_inceptionv3_single5_bs32/32.pkl'
            weight_dir = 'checkpoints1009/ddr512_inceptionv3_noaux_single5_bs32/31.pkl'

        epoch = int(weight_dir.split('/')[-1].split('.')[0])
        checkpoint = torch.load(weight_dir)
        state_dict = checkpoint['net']
        net.load_state_dict(state_dict, strict=True) 

        val_log=open('./logs/test.txt','a')   
        test(net, testloader, optimizer, epoch, val_log)
        exit()

    for epoch in range(start_epoch, start_epoch + args.epochs):
        con_matx.reset()
        net.train()
        total_loss = .0
        total = .0
        correct = .0
        count = .0

        lr = get_dynamic_lr(epoch, args.epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for i, (x, label) in enumerate(dataloader):
            x = x.float().cuda()
            label = label.cuda()
        
            y_pred = net(x)
            con_matx.add(y_pred.detach(),label.detach())
            prediction = y_pred.max(1)[1]
            
            loss = criterion(y_pred, label)

            total_loss += loss.item()
            total += x.size(0)
            correct += prediction.eq(label).sum().item()
            count += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress_bar(i, len(dataloader), 'Loss clf: %.3f | Acc clf: %.3f '
                         % (total_loss / (i + 1), 100. * correct / total))
           
        if (epoch+1)%val_epoch == 0:
            val(net, valloader, optimizer, epoch, test_log)


@torch.no_grad()
def val(net, valloader, optimizer, epoch, test_log):
    global best_acc
    global best_kappa_clf

    net = net.eval()
    total_acc = .0
    total_loss = .0
    correct_clf = .0
    total = .0
    count = .0
    con_matx_clf = meter.ConfusionMeter(args.n_classes)

    prob_clf_list = []
    pred_list = []
    label_clf_list = []


    for i, (x, label_clf) in enumerate(valloader):
        x = x.float().cuda()
        label_clf = label_clf.cuda()

        y_pred = net(x)
        con_matx_clf.add(y_pred.detach(),label_clf.detach())

        _, predicted_clf = y_pred.max(1)

        prob_clf_list.extend(F.softmax(y_pred,dim=-1).cpu().detach().tolist())
        pred_list.extend(predicted_clf.cpu().detach().tolist())
        label_clf_list.extend(label_clf.cpu().detach().tolist())

        total += x.size(0)
        count += 1
        correct_clf += predicted_clf.eq(label_clf).sum().item()

        progress_bar(i, len(valloader), ' Acc clf: %.3f'
                     % (100. * correct_clf / total))

    acc_clf = 100.0*accuracy_score(np.array(label_clf_list), np.array(pred_list))
    kappa_clf = 100.0*cohen_kappa_score(np.array(label_clf_list), np.array(pred_list), weights='quadratic')


    print('val epoch:', epoch, ' val acc clf: ', acc_clf, 'kappa clf: ', kappa_clf)
    test_log.write('Epoch:%d   Accuracy:%.2f   kappa:%.2f   contx:%s \n'%(epoch,acc_clf, kappa_clf, str(con_matx_clf.value())))
    test_log.flush()  

    if kappa_clf > best_kappa_clf:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
        }
        save_name = os.path.join(save_dir, str(epoch) + '.pkl')
        torch.save(state, save_name)
        best_kappa_clf = kappa_clf


@torch.no_grad()
def test(net, testloader, optimizer, epoch, test_log):
    net = net.eval()
    total_acc = .0
    total_loss = .0
    correct_clf = .0
    total = .0
    count = .0
    con_matx_clf = meter.ConfusionMeter(args.n_classes)

    prob_clf_list = []
    pred_list = []
    label_clf_list = []


    for i, (x, label_clf) in enumerate(testloader):
        x = x.float().cuda()
        label_clf = label_clf.cuda()

        y_pred = net(x)
        con_matx_clf.add(y_pred.detach(),label_clf.detach())

        _, predicted_clf = y_pred.max(1)

        prob_clf_list.extend(F.softmax(y_pred,dim=-1).cpu().detach().tolist())
        pred_list.extend(predicted_clf.cpu().detach().tolist())
        label_clf_list.extend(label_clf.cpu().detach().tolist())

        total += x.size(0)
        count += 1
        correct_clf += predicted_clf.eq(label_clf).sum().item()

        progress_bar(i, len(testloader), ' Acc clf: %.3f'
                     % (100. * correct_clf / total))

    acc_clf = 100.0*accuracy_score(np.array(label_clf_list), np.array(pred_list))
    kappa_clf = 100.0*cohen_kappa_score(np.array(label_clf_list), np.array(pred_list), weights='quadratic')


    print('test epoch:%d   acc:%.2f  kappa:%.2f  contx:%s'% (epoch,acc_clf,kappa_clf,str(con_matx_clf.value())))



if __name__ == '__main__':
    main()
