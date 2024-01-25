import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils import data
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(description='Direct SNN Training')

parser.add_argument('--train', action='store_true', default=False,
                    help='train model (default: False)')
parser.add_argument('--load', default=None, metavar='FILE_NAME',
                    help='load model name (default: None)')
parser.add_argument('--save', default=None, metavar='FILE_NAME',
                    help='save model name (default: None)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='set learning rate (default: 1.0)')
parser.add_argument('--ts', type=int, default=8, metavar='TS',
                    help='set time steps (default: 8)')
parser.add_argument('--bs', type=int, default=64, metavar='BS',
                    help='set batch size (default: 64)')
parser.add_argument('--epochs', type=int, default=170, metavar='EPOCHS',
                    help='set training epochs (default: 170)')
parser.add_argument('--dstep', default='80,110,140', metavar='DSTEP',
                    help='set decay steps (default: 80,110,140)')
parser.add_argument('--dgamma', type=float, default=0.1, metavar='DGAMMA',
                    help='set decay factor (default: 0.1)')
parser.add_argument('--opt', default='adadelta', metavar='OPT', choices=['adam', 'adadelta', 'sgd'],
                    help='set optimizer {adam, adadelta, sgd} (default: adadelta)')
parser.add_argument('--a', type=float, default=1.0, metavar='A',
                    help='set gradient delta function width (default: 1.0)')

args = parser.parse_args()
print(args)

TS = args.ts
BS = args.bs
EPOCHS = args.epochs
DECAY_STEPS = [int(s) for s in args.dstep.split(',')]
GAMMA = args.dgamma
LR = args.lr
A = args.a
VTH0 = 1.0
VTH = 0.5

save_model = args.save
load_model = args.load

tStart = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load dataset
train_images = torch.Tensor(np.load('./data/DRINK/train-images.npy'))
train_labels = torch.LongTensor(np.load('./data/DRINK/train-labels.npy'))
test_images = torch.Tensor(np.load('./data/DRINK/test-images.npy'))
test_labels = torch.LongTensor(np.load('./data/DRINK/test-labels.npy'))

train_dataset = data.TensorDataset(train_images, train_labels)
train_loader = data.DataLoader(train_dataset, batch_size=BS, shuffle=True)

test_dataset = data.TensorDataset(test_images, test_labels)
test_loader = data.DataLoader(test_dataset, batch_size=BS, shuffle=False)

class Fire(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, vth):
        ctx.save_for_backward(input, vth)
        return torch.where(input >= vth, torch.ones_like(input), torch.zeros_like(input))

    @staticmethod
    def backward(ctx, grad_output):
        input, vth, = ctx.saved_tensors
        grad_input = torch.where(torch.abs(input - vth) <= A/2., grad_output / A, torch.zeros_like(grad_output))
        grad_vth = torch.where(torch.abs(input - vth) <= A/2., -((grad_output / A)), torch.zeros_like(grad_output)).permute(1, 0, 2, 3).sum(dim=-1).sum(dim=-1).sum(dim=-1).view(1, -1, 1, 1)
        return grad_input, grad_vth

fire = Fire.apply

def reset(s, u):
    return torch.where(torch.eq(s, 1), torch.zeros_like(u), u)

class conv2d_layer(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(conv2d_layer, self).__init__(*kargs, **kwargs)
        self.vth = nn.Parameter(torch.full((self.out_channels, 1, 1), VTH))

    def get_dv(self, input):
        # input size: (ts, bs, in_channels, in_height, in_width)
        # output size: (ts, bs, out_channels, out_height, out_width)
        # weight size: (out_channels, in_channels, kernel_height, kernel_width)

        dv_list = []

        for t in range(TS):
            dv = nn.functional.conv2d(input[t,:,:,:,:], self.weight, None, self.stride, self.padding, self.dilation, self.groups)
            dv_list.append(dv)

        dv = torch.stack(dv_list, dim=0)

        return dv

    def forward(self, input):
        # input size: (bs, ts, in_channels, in_height, in_width)
        # output size: (bs, ts, out_channels, out_height, out_width)
        # weight size: (out_channels, in_channels, kernel_height, kernel_width)

        # Interchange the bs and ts dimension
        input = input.permute(1, 0, 2, 3, 4)

        olist = []

        dv = self.get_dv(input)

        for t in range(TS):
            if t == 0:
                v = dv[t,:,:,:,:]
            else:
                v = v + dv[t,:,:,:,:]

            output = fire(v, self.vth)

            v = reset(output, v)

            olist.append(output)

        output = torch.stack(olist, dim=0)

        # Interchange the bs and ts dimension
        output = output.permute(1, 0, 2, 3, 4)

        return output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.in_channels = 3
        self.num_classes = 6
        self.layer1 = conv2d_layer(in_channels=self.in_channels, out_channels=16, kernel_size=4)
        self.layer2 = conv2d_layer(in_channels=16, out_channels=16, kernel_size=4)
        self.layer3 = conv2d_layer(in_channels=16, out_channels=16, kernel_size=4)
        self.layer4 = conv2d_layer(in_channels=16, out_channels=16, kernel_size=4)
        self.layer5 = conv2d_layer(in_channels=16, out_channels=self.num_classes, kernel_size=4)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.mean(x, dim=(3, 4))
        return x.view(x.size(0), x.size(1), -1)

def encoding(img):
    Sout_list = []
    V_tmp = torch.zeros_like(img)
    for t in range(TS):
        V_tmp = V_tmp + img
        Sout_tmp = torch.sign(torch.sign(V_tmp - VTH0) + 1)
        Sout_list.append(Sout_tmp)
        V_tmp = torch.where(torch.eq(Sout_tmp, 1), V_tmp - VTH0, V_tmp)
    Sout = torch.stack(Sout_list, dim=0).permute(1, 0, 2, 3, 4)
    return Sout

def one_hot_encode(labels, num_classes):
    y = torch.eye(num_classes)
    return y[labels]

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_tstart = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        spike_data = encoding(data)
        output = model(spike_data).permute(1, 0, 2)
        # output = torch.sum(output, dim=0) / TS
        output = output[-1,:]
        # label = one_hot_encode(target, model.module.num_classes).to(device)
        # loss = F.mse_loss(output, label, reduction='mean')
        loss = F.cross_entropy(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Time: {:.2f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), time.time() - train_tstart))
            train_tstart = time.time()

def test(model, device, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            spike_data = encoding(data)
            output = model(spike_data).permute(1, 0, 2)
            output = torch.sum(output, dim=0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(correct, len(test_loader.dataset), accuracy))
    return accuracy

print("Use", torch.cuda.device_count(), "GPUs")
model = nn.DataParallel(Net(), dim=0).to(device)

if load_model != None:
    model.load_state_dict(torch.load(load_model))
    model.eval()

if args.opt == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif args.opt == 'adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=LR)
else:
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

scheduler = MultiStepLR(optimizer, milestones=DECAY_STEPS, gamma=GAMMA)

if args.train:
    print("Training start....")
    acc = test(model, device, test_loader, 0)
    for epoch in range(1, EPOCHS + 1):
        epoch_tstart = time.time()
        print("Learning rate: {}".format(optimizer.param_groups[0]['lr']))

        train(model, device, train_loader, optimizer, epoch)
        test_acc = test(model, device, test_loader, epoch)

        if save_model != None and (test_acc > acc):
            acc = test_acc
            torch.save(model.state_dict(), save_model)
            print("Model saved!")

        scheduler.step()
        print('Time: {:.2f}'.format(time.time() - epoch_tstart))

elif load_model != None:
    print('Evaluate the inference accuracy...')
    test(model, device, test_loader, 0)

tEnd = time.time()
print("Total time: {:f} sec".format(tEnd - tStart))

