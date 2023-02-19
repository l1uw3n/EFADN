import os
import tqdm
import torch as t
import torchvision as tv
from torch.utils.data import DataLoader
from torchvision import transforms
# from torchsummary import summary
import torch.nn.functional as F
# import wandb
from config import opt
from torchnet.meter import AverageValueMeter
from utils.Load_Dataset10 import RandomGenerator,ValGenerator,ImageToImage2D,ImageToImage2Dval
# from evaluate1 import evaluate
from unet import UNet_NonLocal
from torchvision.utils import save_image
import torch.nn as nn
# from tensorboardX import SummaryWriter
from contextual_loss import ContextualLoss
from utils.dice_score import dice_loss

from utils.dice_score import dice_loss
import matplotlib.pyplot as plt
import numpy as np
from torch import optim
import torch
# 计算Prevision系数，即Precison
def calPrecision(gt,pred):
    eps = 0.0001
    P_s = torch.dot(pred.contiguous().view(-1), gt.contiguous().view(-1))
    P_t = torch.sum(pred)
 
    Precision = (P_s.float() + eps)/(P_t.float()+eps)
    return Precision
 
# 计算Recall系数，即Recall
def calRecall(gt,pred):
    eps = 0.0001
    R_s = torch.dot(pred.contiguous().view(-1), gt.contiguous().view(-1))
    R_t = torch.sum(gt)
 
    Recall = (R_s.float()+eps)/(R_t.float()+eps)
    return Recall


def train():
    device = opt.device
    # # 1.预处理数据
    train_tf= transforms.Compose([RandomGenerator()])
    val_tf= transforms.Compose([ValGenerator()])
    train_dataset = ImageToImage2D(train_tf)
    val_dataset = ImageToImage2Dval(val_tf)
    # 2. Create data loaders
    train_loader = DataLoader(train_dataset, 
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.num_workers,
                            pin_memory=True,
                            drop_last=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.num_workers,
                            pin_memory=True,
                            drop_last=True)
    # # 2．初始化网络
    netg = UNet_NonLocal(opt.inputc, opt.n_labels)
    
    # # 2.1判断网络是否已有权重数值
    map_location = lambda storage, loc: storage  # TODO 复习map_location操作
    if opt.netg_path:
        netg.load_state_dict(t.load(opt.netg_path, map_location=map_location))
    # # 2.2 搬移模型到指定设备
    netg.to(device=opt.device)
    
    # # 3. 定义优化策略
    optimize_g = t.optim.Adam(netg.parameters(), lr=opt.lr1, weight_decay=1e-8)
    # optimize_d = t.optim.Adam(netd.parameters(), lr=opt.lr2, weight_decay=1e-8)
    scheduler_g = optim.lr_scheduler.ExponentialLR(optimizer=optimize_g, gamma=0.95)
    # scheduler_d = optim.lr_scheduler.ExponentialLR(optimizer=optimize_d, gamma=0.95)
    criterion_mask = nn.BCEWithLogitsLoss().to(device)
    # criterion_mask = nn.CrossEntropyLoss()
    criterion_grid = ContextualLoss(loss='l1', backbone='resnet50', input_channels=1)
    criterion_grid3 = ContextualLoss(loss='l1', backbone='resnet50', input_channels=3)
    # criterion_grid = nn.MSELoss()
    # criterion_grid3 = nn.MSELoss()
    # # 4. 定义标签, 并且开始注入生成器的输入noise
    errorg_meter = AverageValueMeter()
    # #  6.训练网络
    epochs = range(opt.max_epoch)
    # write = SummaryWriter(log_dir=opt.virs, comment='loss')
    # 6.1 设置迭代
    train_grid_loss = []
    train_img_loss = []
    train_bce_loss = []
    train_dice_loss = []
    test_grid_loss = []
    test_img_loss = []
    test_bce_loss = []
    test_dice_loss = []

    train_grid = []
    train_img = []
    train_bce = []
    train_dice = []

    test_grid = []
    test_img = []
    test_bce = []
    test_dice = []

    train_acc_1 = []
    test_acc_1 = []
    train_acc = []
    test_acc = []

    train_Recall_1 = []
    test_Recall_1 = []
    train_Recall = []
    test_Recall = []

    for epoch in iter(epochs):
        #  6.2 读取每一个batch 数据
        with tqdm.tqdm(total=train_loader.__len__()) as tbar:
            train_bce.clear()
            train_dice.clear()
            train_grid.clear()
            train_img.clear()
            train_acc_1.clear()
            train_Recall_1.clear()
            for ii_,sample in enumerate(train_loader):
                tbar.set_description('Epoch %i' % epoch)
                # print(ii_, sample["imgin"].shape, sample["imgout"].shape, sample["label"].shape)
                # print(true_labels, fake_labels)
                img = sample["img"].to(device=opt.device, dtype=t.float32)
                sst_sla = sample["sst_sla"].to(device=opt.device, dtype=t.float32)
                mask = sample["mask"].to(device=opt.device, dtype=t.long)
                mask = F.one_hot(mask, opt.n_labels).permute(0, 3, 1, 2).float()
                grid = sample["grid"].to(device=opt.device, dtype=t.float32)
                #  6.3开始训练生成器和判别器
                #  注意要使得生成的训练次数小于一些
                netg.train()
                # 训练生成器
                optimize_g.zero_grad()
                fake_img = netg(sst_sla)
                #mask loss
                BCE_loss=criterion_mask(fake_img[0], mask) 
                masks_pred_softmax = F.softmax(fake_img[0], dim=1).float()
                diceloss=dice_loss(masks_pred_softmax,mask.float(),multiclass=True)
                maskloss = BCE_loss + diceloss
                gridloss = criterion_grid(fake_img[2], grid)#*10
                #gan
                ganloss = criterion_grid3(fake_img[1], img)*10
                error_g = ganloss+gridloss+maskloss
                error_g.backward()
                optimize_g.step()
                errorg_meter.add(error_g.item())

                #保存结果

                tbar.set_postfix(lossg=ganloss.item(),lossmask = maskloss.item(), lossgrid = gridloss.item())
                tbar.update(1)

                train_bce.append(BCE_loss.detach().cpu().numpy())
                train_dice.append(diceloss.detach().cpu().numpy())
                train_grid.append(gridloss.detach().cpu().numpy())
                train_img.append(ganloss.detach().cpu().numpy())
                train_acc_1.append(calPrecision(masks_pred_softmax,mask).detach().cpu().numpy())
                train_Recall_1.append(calRecall(masks_pred_softmax,mask).detach().cpu().numpy())

            train_bce_loss.append(sum(train_bce)/len(train_bce))
            train_dice_loss.append(sum(train_dice)/len(train_dice))
            train_grid_loss.append(sum(train_grid)/len(train_grid))
            train_img_loss.append(sum(train_img)/len(train_img))
            train_acc.append(sum(train_acc_1)/len(train_acc_1))
            train_Recall.append(sum(train_Recall_1)/len(train_Recall_1))


            #  7.保存模型
            if (epoch + 1) % opt.save_every == 0:
                netg.eval()
                # ii_,sample = enumerate(val_loader).__next__()
                test_bce.clear()
                test_dice.clear()
                test_grid.clear()
                test_img.clear()
                test_acc_1.clear()
                test_Recall_1.clear() 
                for ii_,sample in enumerate(val_loader):
                    sst_sla = sample["sst_sla"].to(device=opt.device, dtype=t.float32)
                    img = sample["img"].to(device=opt.device, dtype=t.float32)
                    mask = sample["mask"].to(device=opt.device, dtype=t.long)
                    mask = F.one_hot(mask, opt.n_labels).permute(0, 3, 1, 2).float()
                    grid = sample["grid"].to(device=opt.device, dtype=t.float32)
                    fix_fake_image = netg(sst_sla)

                    #mask loss
                    BCE_loss=criterion_mask(fix_fake_image[0], mask) 
                    masks_pred_softmax = F.softmax(fix_fake_image[0], dim=1).float()
                    diceloss=dice_loss(masks_pred_softmax,mask.float(),multiclass=True)
                    maskloss = BCE_loss + diceloss
                    gridloss = criterion_grid(fix_fake_image[2], grid)*10
                    ganloss = criterion_grid3(fix_fake_image[1], img)*10
                    #保存结果
                    tbar.update(1)
                    test_bce.append(BCE_loss.detach().cpu().numpy())
                    test_dice.append(diceloss.detach().cpu().numpy())
                    test_grid.append(gridloss.detach().cpu().numpy())
                    test_img.append(ganloss.detach().cpu().numpy())
                    test_acc_1.append(calPrecision(masks_pred_softmax,mask).detach().cpu().numpy())
                    test_Recall_1.append(calRecall(masks_pred_softmax,mask).detach().cpu().numpy())

                    
                test_bce_loss.append(sum(test_bce)/len(test_bce))
                test_dice_loss.append(sum(test_dice)/len(test_dice))
                test_grid_loss.append(sum(test_grid)/len(test_grid))
                test_img_loss.append(sum(test_img)/len(test_img))
                test_acc.append(sum(test_acc_1)/len(test_acc_1))
                test_Recall.append(sum(test_Recall_1)/len(test_Recall_1))

                plt.figure(figsize=(20, 40))
                plt.subplot(4,2,1)
                plt.imshow(sst_sla[:1][0,:1,...].permute(1, 2, 0).cpu())
                plt.colorbar(extend='both', fraction=0.042, pad=0.04)
                plt.axis('off')
                plt.title('SST')

                plt.subplot(4,2,2)
                plt.imshow(sst_sla[:1][0,1:,...].permute(1, 2, 0).cpu())
                plt.colorbar(extend='both', fraction=0.042, pad=0.04)
                plt.axis('off')
                plt.title('SLA');
                
                plt.subplot(4,2,3)
                plt.imshow(img[:1][0].permute(1, 2, 0).cpu(), cmap='jet')
                plt.colorbar(extend='both', fraction=0.042, pad=0.04)
                plt.axis('off')
                plt.title('img')

                plt.subplot(4,2,4)
                plt.imshow(np.squeeze(grid[:1][0].cpu()), cmap='jet')
                plt.colorbar(extend='both', fraction=0.042, pad=0.04)
                plt.axis('off')
                plt.title('grid')

                plt.subplot(4,2,5)
                plt.imshow(fix_fake_image[1].data[:1][0].permute(1, 2, 0).cpu(), cmap='jet')
                plt.colorbar(extend='both', fraction=0.042, pad=0.04)
                plt.axis('off')
                plt.title('preimg')

                plt.subplot(4,2,6)
                plt.imshow(fix_fake_image[2].data[:1][0].permute(1, 2, 0).cpu(), cmap='jet')
                plt.colorbar(extend='both', fraction=0.042, pad=0.04)
                plt.axis('off')
                plt.title('pregrid')

                plt.subplot(4,2,7)
                plt.imshow(mask[:1][0].cpu().argmax(0))
                plt.colorbar(extend='both', fraction=0.042, pad=0.04)
                plt.axis('off')
                plt.title('mask')

                plt.subplot(4,2,8)
                plt.imshow(fix_fake_image[0].data[:1][0].cpu().argmax(0))
                plt.colorbar(extend='both', fraction=0.042, pad=0.04)
                plt.axis('off')
                plt.title('prmask')
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.tight_layout()
                plt.savefig("%s/%s.png" % (opt.save_path, epoch))
                plt.close()
                errorg_meter.reset()

                dict = {
                    "test_bce_loss" : test_bce_loss,
                    "test_dice_loss" : test_dice_loss,
                    "test_grid_loss" : test_grid_loss,
                    "test_img_loss" : test_img_loss,
                    "train_bce_loss" : train_bce_loss,
                    "train_dice_loss" : train_dice_loss,
                    "train_grid_loss" : train_grid_loss,
                    "train_img_loss" : train_img_loss,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "train_recall": train_Recall,
                    "test_recall": test_Recall
                    
                }
                np.save(opt.save_path+'/save_%s' % epoch, dict)

            if (epoch + 1) % opt.save_model == 0:
                t.save(netg.state_dict(), opt.model_save+'/netg_%s.pth' % epoch)
        scheduler_g.step()
train()