# -*- coding: utf-8 -*-
__author__ = 'huangyf'
import os
import time
import math
import torch
import scipy.misc
import numpy as np
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch_einsum import einsum
from tensorboardX import SummaryWriter

use_cuda = torch.cuda.is_available()

batch_size = 100

train_data = datasets.MNIST(root='./pytorch/mnist_data', train=True, transform=transforms.ToTensor(), download=True)
test_data = datasets.MNIST(root='./pytorch/mnist_data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_data, batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size, shuffle=False)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since):
    now = time.time()
    s = now - since
    return '%s ' % (asMinutes(s))


def save_images(images, size, path):
    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]

    merge_img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j * h:j * h + h, i * w:i * w + w] = image

    return scipy.misc.imsave(path, merge_img)


class CapsNet(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, out_channel2, kernel_size2, stride2,
                 capsule_length, feat_size, n_classes, r, input_size):
        super(CapsNet, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channel2 = out_channel2
        self.kernel_size2 = kernel_size2
        self.stride2 = stride2
        self.capsule_length = capsule_length
        self.feat_size = feat_size
        self.n_classes = n_classes
        self.input_size = input_size
        self.routing = routing(self.out_channel2 * (((self.input_size - self.kernel_size + 1) //
                                                     self.stride - self.kernel_size2 + 1) // self.stride2) ** 2,
                               self.n_classes, r=r)
        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, self.kernel_size, stride)
        self.pri_caps = nn.ModuleList(
            [nn.Conv2d(self.out_channel, self.out_channel2, self.kernel_size2, stride2) for _ in
             range(self.capsule_length)])

        self.capsule_weight = nn.Parameter(
            2 * (torch.randn(self.out_channel2 * (((self.input_size - self.kernel_size + 1) //
                                                   self.stride - self.kernel_size2 + 1) // self.stride2) ** 2,
                             self.capsule_length, self.feat_size, self.n_classes) - 0.5))
        # if use_cuda:
        #     self.capsule_weight = self.capsule_weight.cuda()

        self.fc = nn.Sequential(nn.Linear(feat_size, 512),
                                nn.ReLU(inplace=True),
                                nn.Linear(512, 1024),
                                nn.ReLU(inplace=True),
                                nn.Linear(1024, 784),
                                nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        ## cap(x):(batch_size,out_channel,h,w)(100,32,6,6)
        caps = [cap(x) for cap in self.pri_caps]
        ## caps:(batch_size,cap_length,out_channel,h,w)(100,8,32,6,6)
        caps = torch.stack(caps, 1)
        caps = caps.view(batch_size, self.capsule_length, -1)
        caps = self.routing.squashing(caps)
        ##top_caps:(batch_size,...,feat_size,n_classes)(100,1152,16,10)
        top_caps = einsum('imj,jmkl->ijkl', caps, self.capsule_weight)
        V = self.routing.foward(top_caps)
        return V

    def reconstruction(self, digit_caps, imgs, labels=None, train=True):
        ## labels:numpy(batch_size)
        digit_caps = torch.transpose(digit_caps, 1, 2)
        ## (batch_size,16,10)-->(batch_size,10,16)
        if train:
            activ_vecs = torch.stack([digit_caps[i, labels[i], :] for i in range(len(labels))]).squeeze(1)
        else:
            idx = digit_caps.norm(p=2, dim=2).topk(1)[1].data.squeeze(1).cpu().numpy()
            activ_vecs = torch.stack([digit_caps[i, idx[i], :] for i in range(len(idx))]).squeeze(1)
        re_imgs = self.fc(activ_vecs)
        loss = torch.sum((re_imgs - imgs.view(imgs.size()[0], -1)) ** 2, dim=1).mean()
        if train:
            return loss
        else:
            return re_imgs.view(imgs.size()[0], 28, 28), loss


class routing:
    def __init__(self, in_caps, out_caps, r):
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.r = r
        # self.b = Variable(torch.zeros(batch_size, out_caps, in_caps),requires_grad=False)

    def foward(self, U):
        ## u_size=(batch_size,...,feat_size,n_classes)(100,1152,16,10)
        ## b:(n_classes,..)(10,1152)
        b = Variable(torch.zeros(self.out_caps, self.in_caps), requires_grad=False)
        if use_cuda:
            b = b.cuda()
        for i in range(self.r):
            C = softmax(b, -1)
            ## S:(batch_size,feat_size,n_classes)(100,1152,16,10)->(100,16,10),逐元素相乘
            S = einsum('ijkl,lj->ijkl', U, C).sum(dim=1)
            V = self.squashing(S)
            b = b + einsum('ikmj,imlj->ijkl', U, V.unsqueeze(2)).mean(dim=0).squeeze(-1)
        return V

    def squashing(self, S):
        ## S_size:(batch_size,out_caps,feat_size)
        S_norm = S.norm(p=2, dim=-1).unsqueeze(-1)
        return S_norm ** 2 / (1 + S_norm ** 2) * (S / S_norm)


def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)))
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def margin_loss(input, target, m1, m2, lam):
    # input,target:(batch_size,n_classes)
    zero = Variable(torch.zeros(*target.size()))
    if use_cuda:
        zero = zero.cuda()
    max1 = torch.max(zero, m1 - target) ** 2
    max2 = torch.max(zero, target - m2) ** 2
    loss = input * max1 + lam * (1 - input) * max2
    return torch.sum(loss, 1).mean()


if __name__ == "__main__":
    epochs = 30
    lr = 0.001
    n_classes = 10
    caps_net = CapsNet(in_channel=1, out_channel=256, kernel_size=9, stride=1,
                       out_channel2=32, kernel_size2=9, stride2=2, capsule_length=8, feat_size=16, n_classes=10, r=3,
                       input_size=28)
    if use_cuda:
        caps_net = caps_net.cuda()
    optimizer = optim.Adam(caps_net.parameters(), lr=lr)
    print(caps_net)
    print('num parameters:{}'.format(sum(param.numel() for param in caps_net.parameters())))

    tb = SummaryWriter()

    if os.path.exists('cap_net_rec'):
        with open('cap_net_rec', 'rb') as f:
            caps_net = torch.load('cap_net_rec')
            print('load cap_net_rec yet.')

    start = time.time()
    loss_train = []
    accuracy_train = []
    accuracy_test = []
    for epoch in range(epochs):
        caps_net.train()
        mini_size = 60000 // batch_size
        loss_all = []
        accuracy_all = []
        correct_prediction = 0.0
        total_counter = 0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs = Variable(imgs)
            labels_onehot = torch.FloatTensor(batch_size, n_classes)
            labels_onehot.zero_()
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = Variable(labels_onehot)
            if use_cuda:
                imgs = imgs.cuda()
                labels_onehot = labels_onehot.cuda()
            optimizer.zero_grad()
            output = caps_net.forward(imgs)

            loss1 = margin_loss(labels_onehot, output.norm(p=2, dim=1), 0.9, 0.1, 0.5)
            ## add reconstruction error
            loss2 = caps_net.reconstruction(output, imgs, labels, train=True)
            loss = loss1 + 0.0005 * loss2

            loss.backward()
            optimizer.step()
            predictions = output.norm(p=2, dim=1).topk(1)[1].data.squeeze(1).cpu().numpy()
            total_counter += batch_size
            correct_prediction += np.sum(labels.numpy() == predictions)
            accuracy = correct_prediction / total_counter
            loss_all.append(loss.data[0])
            accuracy_all.append(accuracy)
            if i % 100 == 0:
                loss_train.extend(loss_all)
                accuracy_train.extend(accuracy_all)
                tb.add_scalar('loss', loss.data[0], epoch * mini_size + i)
                tb.add_scalar('accuracy_train', accuracy, epoch * mini_size + i)
                print(
                    'epoch:{} [{}/{}] time:{} loss:{:.4f},accuracy:{:.4f}'.format(epoch + 1, i, mini_size,
                                                                                  timeSince(start),
                                                                                  sum(loss_all) / len(loss_all),
                                                                                  accuracy_all[-1]))
        caps_net.eval()
        correct_prediction = 0.0
        total_counter = 0

        for j, (imgs, labels) in enumerate(test_loader):
            imgs = Variable(imgs)
            labels_onehot = torch.FloatTensor(batch_size, n_classes)
            labels_onehot.zero_()
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = Variable(labels_onehot)
            if use_cuda:
                imgs = imgs.cuda()
                labels_onehot = labels_onehot.cuda()
            output = caps_net.forward(imgs)
            rec_imgs, loss = caps_net.reconstruction(output, imgs, train=False)
            predictions = output.norm(p=2, dim=1).topk(1)[1].data.squeeze(1).cpu().numpy()
            correct_prediction += np.sum(labels.numpy() == predictions)
            total_counter += batch_size
            accuracy_test.append(correct_prediction / (total_counter))
            if j % 100 == 0 and j != 0:
                tb.add_scalar('accuracy_test', correct_prediction / (total_counter), epoch)
                break
        ## please set epochs=1 to save the figure
        save_images(imgs.cpu().data.numpy()[:, 0, :, :], [10, 10], 'test.png')
        save_images(rec_imgs.cpu().data.numpy(), [10, 10], 'test_rec.png')
        print('epoch:{} test_accuracy:{}'.format(epoch + 1, correct_prediction / (total_counter)))
    np.savetxt('loss_train.txt', loss_train)
    np.savetxt('accuracy_train.txt', accuracy_train)
    np.savetxt('accuracy_test.txt', accuracy_test)
    torch.save(caps_net, 'cap_net_rec')
    print('Save the model')
    tb.close()

    ## 分量的改变的重构
    for k in range(100):
        imgs_changes = np.zeros((16, 11, 28, 28))
        for i in range(16):
            for j, change in enumerate(np.linspace(-0.25, 0.25, 11)):
                matrix = Variable(torch.zeros(16, 10)).cuda()
                matrix[i, :] = change
                rec_imgs_i = \
                caps_net.reconstruction((output[k] + matrix).unsqueeze(0), imgs[k].unsqueeze(0), train=False)[0]
                imgs_changes[i, j] = rec_imgs_i.cpu().data.numpy()[0, :, :]
        imgs_changes = imgs_changes.reshape(-1, 28, 28)
        save_images(imgs_changes, [16, 11], './capsule/changes{}.png'.format(k))
