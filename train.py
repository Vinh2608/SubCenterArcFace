import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import torchvision
from torchvision import datasets, transforms
from torchvision import models
from models.mobilefacenet import MobileFaceNet
from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from dataset import Dataset
from config import Config
from models import *

best_loss = 39.594764709472656
s = 64
m = 0.2
def train(model, loss_func, device, train_loader, optimizer, loss_optimizer, epoch):
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
        if batch_idx % 100 == 0:
            print("Epoch {} Iteration {}: Loss = {}".format(epoch, batch_idx, loss))
            if loss < best_loss:
                best_loss = loss
                PATH = f'/content/SubCenterArcFace/checkpoints/{config.model}_model_s={s}_m={m}_{best_loss}_{epoch}_arcfaceloss.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                }, PATH)


### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, test_labels, train_embeddings, train_labels, False
    )
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))


if __name__ == '__main__':
    global s
    global m
    device = torch.device("cuda")
    config = Config()
    if config.model == 'resnet101':
        model = resnet101(pretrained = True)
        model = model.to(device)
    elif config.model == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
        model_dict = model.state_dict()
        pretrained_dict = torch.load(opt.load_model_path)
        pretrained_dict =  {k:v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        for params in model.parameters():
            params.trainable =  False
        model.fc5.trainable = True
        model.bn5.trainable = True
    else:
        model = MobileFaceNet(512).to(device)
        # model_dict = model.state_dict()
        # pretrained_dict = torch.load('/content/mobilefacenet_s=64_m=0.2batch_size=200_align_frontal__70_acc905.pth')
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

    train_dataset = Dataset('VN-celeb_align_frontal_full', 'label_train.txt', phase='train', input_shape=(3, 128, 128))
    test_dataset = Dataset('VN-celeb_align_frontal_full', 'label_test.txt', phase='test', input_shape=(3, 128, 128))
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=200,
                                   shuffle=True,
                                   num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    initial_epoch = 1
    num_epochs = 100

    ### pytorch-metric-learning stuff ###
    loss_func = losses.ArcFaceLoss(num_classes=1021, embedding_size=512, margin=0.2, scale=64).to(device)
    loss_optimizer = torch.optim.Adam(loss_func.parameters(), lr=1e-4)
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
    ### pytorch-metric-learning stuff ###
    for epoch in range(initial_epoch, initial_epoch+ num_epochs + 1):
        train(model, loss_func, device, train_loader, optimizer, loss_optimizer, epoch)
        test(train_dataset, test_dataset, model, accuracy_calculator)
