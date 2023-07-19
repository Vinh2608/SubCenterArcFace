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

best_loss = 10
s = 64
m = 0.2
best_acc = 0.4
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
                PATH = f'/content/SubCenterArcFace/checkpoints/{config.model}_model_s={s}_m={m}_{best_loss}_{epoch}_arcfaceloss73.pt'
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
    global best_acc
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, test_labels, train_embeddings, train_labels, False
    )
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))
    if accuracies["precision_at_1"] > best_acc:
        best_acc = accuracies["precision_at_1"]
        PATH = f'/content/SubCenterArcFace/checkpoints/{config.model}_model_s={s}_m={m}_{best_loss}_acc{best_acc}_{epoch}_arcfaceloss73.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_loss,
        }, PATH)


if __name__ == '__main__':
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
                                   batch_size=16,
                                   shuffle=True,
                                   num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=50)

    optimizer = optim.Adam(model.parameters(), lr=0.1)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    initial_epoch = 0
    num_epochs = 100

    model.to(device)
    ### pytorch-metric-learning stuff ###
    loss_func = losses.SubCenterArcFaceLoss(num_classes=1021, embedding_size=512, margin=m, scale=m).to(device)
    loss_optimizer = torch.optim.Adam(loss_func.parameters(), lr=1e-4)
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
    ### pytorch-metric-learning stuff ###
    for epoch in range(initial_epoch, initial_epoch+ num_epochs + 1):
        train(model, loss_func, device, train_loader, optimizer, loss_optimizer, epoch)
        test(train_dataset, test_dataset, model, accuracy_calculator)
    train_embeddings, train_labels = get_all_embeddings(train_dataset, model)    
    outliers, _ = loss_func.get_outliers(train_embeddings, train_labels.squeeze(1))
    print(f"There are {len(outliers)} outliers")
    torch.save({'outlier': outliers},'/content/SubCenterArcFace/outliers/outlier.pt' )