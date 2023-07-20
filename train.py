import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import torchvision
from torchvision import datasets, transforms
from torchvision import models
#from models.mobilefacenet import MobileFaceNet
from models import *
from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from dataset import Dataset
from config import Config
from models import *
import os
from datetime import datetime
import time
import numpy as np

best_loss = 10
s = 64
m = 0.2
best_acc = 0.4
def train(model, loss_func, device, train_loader, optimizer, loss_optimizer, epoch, log_fil2):
    global best_loss
    global s
    global m
    config = Config()
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        loss_optimizer.zero_grad()
        embeddings = model(data)
        loss = loss_func(embeddings, labels)
        loss.backward()
        optimizer.step()
        loss_optimizer.step()
        time_str = time.asctime(time.localtime(time.time()))

        log_file2.write('{} train epoch {} iter {} loss {}\n'.format(time_str, epoch, batch_idx, loss))
        if batch_idx % 100 == 0:
            print("Epoch {} Iteration {}: Loss = {}".format(epoch, batch_idx, loss))
            if loss < best_loss:
                best_loss = loss
                PATH = f'/content/SubCenterArcFace/checkpoints/{config.model}_model_s={s}_m={m}_{best_loss}_{epoch}_subcenter_train.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_optimizer_state_dict': loss_optimizer.state_dict(),
                    #'loss': list(loss_func.parameters())
                }, PATH)

### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)

### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(train_set, test_set, model, accuracy_calculator,epoch,log_file1):
    global best_acc
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, test_labels, train_embeddings, train_labels, False
    )
    print("Test set accuracy at epoch {} (Precision@1) = {}".format(epoch,accuracies["precision_at_1"]))
    time_str = time.asctime(time.localtime(time.time()))
    log_file1.write('{} test epoch {} acc {}\n'.format(time_str, epoch, accuracies["precision_at_1"]))


if __name__ == '__main__':
    runtime = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_file1 = open(os.path.join('logs', '_s=' + str(s) + '_m=' + str(m) + "subcenter_test_r34.txt"), "w", encoding="utf-8")
    log_file2 = open(os.path.join('logs', '_s=' + str(s) + '_m=' + str(m) + "subcenter_train_r34.txt"), "w", encoding="utf-8")


    device = torch.device("cuda")
    config = Config()
    checkpoint = torch.load(config.load_model_checkpoint)
    if config.model == 'resnet101':
        model = resnet101(pretrained = True)
        model = model.to(device)
    elif config.model == 'resnet18':
        model = resnet_face18(use_se=config.use_se)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(config.load_model_checkpoint)
        pretrained_dict =  {k:v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        for params in model.parameters():
            params.trainable =  False
        model.fc5.trainable = True
        model.bn5.trainable = True
        model = model.to(device)
    elif config.model == 'iresnet50':
        model = get_model("r50", fp16=False)
        model.load_state_dict(checkpoint)
    elif config.model == 'iresnet34':
        model = get_model("r34", fp16=False)
        model.load_state_dict(checkpoint)
    else:
        model = MobileFaceNet(512).to(device)
        # model_dict = model.state_dict()
        # pretrained_dict = checkpoint['model_state_dict']
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)
    
    # model.conv1.requires_grad = False
    # model.conv2_dw.requires_grad_ = False
    # model.conv_23.requires_grad = False
    # model.conv_3.requires_grad = False
    # model.conv_34.requires_grad = False
    # model.conv_4.requires_grad = False
    # model.conv_45.requires_grad = False
    # model.conv_5.requires_grad = False
    # model.conv_6_dw.requires_grad = True
    # model.linear.requires_grad = True
    # model.bn.requires_grad = True

    train_dataset = Dataset('VN-celeb_align_frontal_full', 'label_train.txt', phase='train', input_shape=(3, 112, 112))
    test_dataset = Dataset('VN-celeb_align_frontal_full', 'label_test.txt', phase='test', input_shape=(3, 112, 112))
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=100,
                                   shuffle=True,
                                   num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50)

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    initial_epoch = 0 #checkpoint['epoch']
    num_epochs = 100

    model.to(device)
    ### pytorch-metric-learning stuff ###
    loss_func = losses.SubCenterArcFaceLoss(num_classes=1021, embedding_size=512, margin=m, scale=m).to(device)
    loss_optimizer = torch.optim.Adam(loss_func.parameters(), lr=1e-2)
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
    ### pytorch-metric-learning stuff ###
    for epoch in range(initial_epoch, initial_epoch + num_epochs + 1):
        train(model, loss_func, device, train_loader, optimizer, loss_optimizer, epoch,log_file2)
        test(train_dataset, test_dataset, model, accuracy_calculator,epoch, log_file1)
    
    log_file1.close()
    log_file2.close()
    
    train_embeddings, train_labels = get_all_embeddings(train_dataset, model)    
    outliers1, _ = loss_func.get_outliers(train_embeddings, train_labels.squeeze(1))
    print(f"There are {len(outliers1)} outliers")
    
    train_imgs = train_dataset.imgs
    for i in range(len(outliers1)):
        train_imgs = np.delete(train_imgs,outliers1[i])

    test_embeddings, test_labels = get_all_embeddings(test_dataset, model)
    outliers2, _ = loss_func.get_outliers(test_embeddings, test_labels.squeeze(1))
    print(f"There are {len(outliers2)} outliers")

    test_imgs = test_dataset.imgs
    for i in range(len(outliers2)):
        test_imgs = np.delete(test_imgs, outliers2[i])    
        
    torch.save({
        'train_dataset_ls': train_dataset.imgs,
        'test_dataset_ls': test_dataset.imgs
        },'/content/SubCenterArcFace/outliers/train_test_remove_outlier.pt' )