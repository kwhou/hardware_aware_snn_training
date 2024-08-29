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
                    help='set training epochs (default: 120)')
parser.add_argument('--dstep', default='80,110,140', metavar='DSTEP',
                    help='set decay steps (default: 80,110,140)')
parser.add_argument('--dgamma', type=float, default=0.1, metavar='DGAMMA',
                    help='set decay factor (default: 0.1)')
parser.add_argument('--opt', default='adadelta', metavar='OPT', choices=['adam', 'adadelta', 'sgd'],
                    help='set optimizer {adam, adadelta, sgd} (default: adadelta)')
parser.add_argument('--a', type=float, default=1.0, metavar='A',
                    help='set gradient delta function width (default: 1.0)')
parser.add_argument('--vmod', default='ideal', metavar='MODE', choices=['ideal', 'global', 'local', 'both'],
                    help='set variation mode {ideal, global, local, both} (default: ideal)')
parser.add_argument('--corner', default='tt', metavar='CORNER', choices=['tt', 'ff', 'ss', 'fs', 'sf'],
                    help='set corner of global variation {tt, ff, ss, fs, sf} (default: tt)')
parser.add_argument('--range', default='50,200', metavar='RANGE',
                    help='set local variation percentage range (default: 50,200)')

args = parser.parse_args()
print(args)

TS = args.ts
BS = args.bs
EPOCHS = args.epochs
DECAY_STEPS = [int(s) for s in args.dstep.split(',')]
GAMMA = args.dgamma
LR = args.lr
A = args.a
VMOD = args.vmod
CORNER = args.corner

VTH = 0.5
VTH_QMAX = 32
VTH_QMIN = 8
V_QMAX = 32
V_QMIN = -32

VMAX = 0.8
VR_IDEAL = 0.4
VR_TT = 0.39975
VR_FS = 0.34787
VR_SF = 0.45182
VR_FF = 0.38369
VR_SS = 0.41550

VRQ_TT = round((VR_TT - VR_IDEAL) * V_QMAX / (VMAX - VR_IDEAL))
VRQ_FS = round((VR_FS - VR_IDEAL) * V_QMAX / (VMAX - VR_IDEAL))
VRQ_SF = round((VR_SF - VR_IDEAL) * V_QMAX / (VMAX - VR_IDEAL))
VRQ_FF = round((VR_FF - VR_IDEAL) * V_QMAX / (VMAX - VR_IDEAL))
VRQ_SS = round((VR_SS - VR_IDEAL) * V_QMAX / (VMAX - VR_IDEAL))
VRQ = 0

RAND_LOW = int(args.range.split(',')[0])
RAND_HIGH = int(args.range.split(',')[1]) + 1

save_model = args.save
load_model = args.load

tStart = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load dataset
train_images = torch.Tensor(np.load('./dataset/DRINK/train-images.npy'))
train_labels = torch.LongTensor(np.load('./dataset/DRINK/train-labels.npy'))
test_images = torch.Tensor(np.load('./dataset/DRINK/test-images.npy'))
test_labels = torch.LongTensor(np.load('./dataset/DRINK/test-labels.npy'))

train_dataset = data.TensorDataset(train_images, train_labels)
train_loader = data.DataLoader(train_dataset, batch_size=BS, shuffle=True)

test_dataset = data.TensorDataset(test_images, test_labels)
test_loader = data.DataLoader(test_dataset, batch_size=BS, shuffle=False)

if CORNER == 'ff':
    dv_table = torch.tensor(np.load('./dv_table/dv_table_ff.npy'), device=device)
elif CORNER == 'ss':
    dv_table = torch.tensor(np.load('./dv_table/dv_table_ss.npy'), device=device)
elif CORNER == 'fs':
    dv_table = torch.tensor(np.load('./dv_table/dv_table_fs.npy'), device=device)
elif CORNER == 'sf':
    dv_table = torch.tensor(np.load('./dv_table/dv_table_sf.npy'), device=device)
else:
    dv_table = torch.tensor(np.load('./dv_table/dv_table_tt.npy'), device=device)

def compute_threshold(w):
    # weight size: (out_channels, in_channels, kernel_height, kernel_width)
    out = torch.mean(torch.abs(w), dim=[1,2,3], keepdim=True) * 0.7
    return out

def compute_alpha(w):
    # weight size: (out_channels, in_channels, kernel_height, kernel_width)
    threshold = compute_threshold(w)
    w_abs = torch.abs(w)
    w_sum = torch.sum(torch.where(torch.gt(w_abs, threshold), w_abs, torch.zeros_like(w)), dim=[1,2,3], keepdim=True)
    n = torch.sum(torch.where(torch.gt(w_abs, threshold), torch.ones_like(w), torch.zeros_like(w)), dim=[1,2,3], keepdim=True)
    alpha = torch.div(w_sum, n)
    return alpha

def ternarize(w):
    # weight size: (out_channels, in_channels, kernel_height, kernel_width)
    threshold = compute_threshold(w)
    out = torch.sign(torch.add(torch.sign(torch.add(w, threshold)), torch.sign(torch.add(w, -threshold))))
    return out

def vth_quantize(vth, alpha, VTH_QMIN, VTH_QMAX):
    # type: (Tensor, Tensor, int, int) -> Tensor
    if args.train:
        return (vth / alpha).round().clamp(min=VTH_QMIN, max=VTH_QMAX) * alpha
    else:
        return (vth / alpha).round().clamp(min=VTH_QMIN, max=VTH_QMAX)

def v_clamp(v, alpha, V_QMIN, V_QMAX):
    # type: (Tensor, Tensor, int, int) -> Tensor
    if args.train:
        return (v / alpha).clamp(min=V_QMIN, max=V_QMAX) * alpha
    else:
        return v.clamp(min=V_QMIN, max=V_QMAX)

def weight_compare(weight):
    # weight size: (out_channels, in_channels, kernel_height, kernel_width)
    we = torch.where(weight == 1, torch.ones_like(weight), torch.zeros_like(weight))
    wi = torch.where(weight == -1, torch.ones_like(weight), torch.zeros_like(weight))
    w0 = torch.where(weight == 0, torch.ones_like(weight), torch.zeros_like(weight))
    return we, wi, w0

def dv_lookup(ne, ni, n0, dv_table, VMOD, RAND_LOW, RAND_HIGH, VMAX, VR_IDEAL, V_QMAX):
    # type: (Tensor, Tensor, Tensor, Tensor, str, int, int, float, float, int) -> Tensor
    # input size: (bs, ts, out_channels, in_channels, kernel_height, kernel_width)
    # output size: (bs, ts, out_channels, in_channels, kernel_height, kernel_width)
    # Apply global variation
    if VMOD == 'global' or VMOD == 'both':
        out = dv_table[ne, ni, n0] * V_QMAX / ((VMAX - VR_IDEAL) * 1000)
    else:
        out = ne - ni
    if VMOD == 'local' or VMOD == 'both':
        out = out * (torch.randint(RAND_LOW, RAND_HIGH, out.shape, device=device) / 100)
    return out

def fire_forward(input, vth):
    # input size: (bs, out_channels, out_height, out_width)
    # vth size: (out_channels, 1, 1)
    with torch.no_grad():
        return torch.where(input >= vth, torch.ones_like(input), torch.zeros_like(input))

def fire_backward(grad, input, vth, A):
    # type: (Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor, None]
    # grad size: (bs, out_channels, out_height, out_width)
    # input size: (bs, out_channels, out_height, out_width)
    # vth size: (out_channels, 1, 1)
    grad_input = torch.where(torch.abs(input - vth) <= A/2, grad / A, torch.zeros_like(grad))
    grad_vth = torch.where(torch.abs(input - vth) <= A/2, -(grad / A), torch.zeros_like(grad)).sum(dim=[0,2,3]).view(-1, 1, 1)
    return grad_input, grad_vth, None

class Fire(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, vth, alpha):
        vq = v_clamp(input, alpha, V_QMIN, V_QMAX)
        vthq = vth_quantize(vth, alpha, VTH_QMIN, VTH_QMAX)
        ctx.save_for_backward(vq, vthq)
        return fire_forward(vq, vthq)

    @staticmethod
    def backward(ctx, grad_output):
        input, vth = ctx.saved_tensors
        return fire_backward(grad_output, input, vth, A)

fire = Fire.apply

def reset(s, u):
    return torch.where(torch.eq(s, 1), torch.zeros_like(u), u)

def my_conv_forward(input, weight, stride, dv_table, VMOD, RAND_LOW, RAND_HIGH, VMAX, VR_IDEAL, V_QMAX):
    # type: (Tensor, Tensor, int, Tensor, str, int, int, float, float, int) -> Tensor
    # input size: (bs, ts, in_channels, in_height, in_width)
    # weight size: (out_channels, in_channels, kernel_height, kernel_width)
    # output size: (bs, ts, out_channels, out_height, out_width)

    with torch.no_grad():
        batch_size = input.size(0)
        time_steps = input.size(1)
        in_channels = input.size(2)
        in_height = input.size(3)
        in_width = input.size(4)
        kernel_height = weight.size(2)
        kernel_width = weight.size(3)
        out_channels = weight.size(0)
        out_height = (in_height - kernel_height) // stride + 1
        out_width = (in_width - kernel_width) // stride + 1

        wt = ternarize(weight)
        alpha = compute_alpha(weight)

        output = torch.zeros(batch_size, time_steps, out_channels, out_height, out_width, device=input.device)

        we, wi, w0 = weight_compare(wt)

        for i in range(time_steps):
            ne = nn.functional.conv2d(input[:,i,:,:,:], we).long()
            ni = nn.functional.conv2d(input[:,i,:,:,:], wi).long()
            n0 = nn.functional.conv2d(input[:,i,:,:,:], w0).long()
            dv = dv_lookup(ne, ni, n0, dv_table, VMOD, RAND_LOW, RAND_HIGH, VMAX, VR_IDEAL, V_QMAX)
            output[:,i,:,:,:] = dv

        if args.train:
            output = output  * alpha.view(-1,1,1)

        return output

def my_conv_backward(grad, input, weight, stride):
    # type: (Tensor, Tensor, Tensor, int) -> Tuple[Tensor, Tensor, None]
    # grad size: (bs, ts, out_channels, out_height, out_width)
    # input size: (bs, ts, in_channels, in_height, in_width)
    # weight size: (out_channels, in_channels, kernel_height, kernel_width)

    batch_size = input.size(0)
    time_steps = input.size(1)
    in_channels = input.size(2)
    kernel_height = weight.size(2)
    kernel_width = weight.size(3)
    out_channels = weight.size(0)
    out_height = grad.size(3)
    out_width = grad.size(4)

    wt = ternarize(weight)
    alpha = compute_alpha(weight)

    grad_input = torch.zeros_like(input)
    grad_weight = torch.zeros_like(weight)
    for t in range(time_steps):
        grad_input[:,t,:,:,:] = torch.nn.grad.conv2d_input(input[:,t,:,:,:].shape, alpha * wt, grad[:,t,:,:,:], stride)
        grad_weight += torch.nn.grad.conv2d_weight(input[:,t,:,:,:], weight.shape, grad[:,t,:,:,:], stride)

    return grad_input, grad_weight, None

class MyConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, stride):
        ctx.save_for_backward(input, weight)
        ctx.stride = stride
        return my_conv_forward(input, weight, stride, dv_table, VMOD, RAND_LOW, RAND_HIGH, VMAX, VR_IDEAL, V_QMAX)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        return my_conv_backward(grad_output, input, weight, ctx.stride)

my_conv = MyConv.apply

class conv2d_layer(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(conv2d_layer, self).__init__(*kargs, **kwargs)
        self.vth = nn.Parameter(torch.full((self.out_channels, 1, 1), VTH))

    def forward(self, input):
        # input size: (bs, ts, in_channels, in_height, in_width)
        # output size: (bs, ts, out_channels, out_height, out_width)
        # weight size: (out_channels, in_channels, kernel_height, kernel_width)

        dv = my_conv(input, self.weight, self.stride[0])

        olist = []
        for t in range(TS): # TS: time step (default 16)
            if t == 0:
                if VMOD == 'global' or VMOD == 'both':
                    v = torch.full_like(dv[:,t,:,:,:], VRQ, device=device) + dv[:,t,:,:,:]
                else:
                    v = dv[:,t,:,:,:]
            else:
                v = v + dv[:,t,:,:,:]
            output = fire(v, self.vth, compute_alpha(self.weight).view(-1,1,1))
            v = reset(output, v)
            olist.append(output)

        output = torch.stack(olist, dim=1)

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
        return x

def encoding(img, TS):
    # type: (Tensor, int) -> Tensor
    Sout_list = []
    V_tmp = torch.zeros_like(img)
    for t in range(TS):
        V_tmp = V_tmp + img
        Sout_tmp = torch.sign(torch.sign(V_tmp - 1) + 1)
        Sout_list.append(Sout_tmp)
        V_tmp = torch.where(torch.eq(Sout_tmp, 1), V_tmp - 1, V_tmp)
    Sout = torch.stack(Sout_list, dim=1)
    return Sout

def one_hot_encode(labels, num_classes, device):
    y = torch.eye(num_classes).to(device)
    return y[labels]

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_tstart = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        spike_data = encoding(data, TS)
        output = model(spike_data).permute(1, 0, 2)
        output = torch.sum(output, dim=0) / TS
        label = one_hot_encode(target, model.module.num_classes, device)
        loss = F.mse_loss(output, label, reduction='mean')

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
    test_tstart = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            spike_data = encoding(data, TS)
            output = model(spike_data).permute(1, 0, 2)
            output = torch.sum(output, dim=0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        print('Time: {:.2f}'.format(time.time() - test_tstart))
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(correct, len(test_loader.dataset), accuracy))
    return accuracy

print("Use", torch.cuda.device_count(), "GPUs")
model = nn.DataParallel(Net(), dim=0).to(device)

if load_model != None:
    model.load_state_dict(torch.load(load_model))
    model.eval()

if args.opt == "adam":
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif args.opt == "adadelta":
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
        
        if epoch < int(EPOCHS*0.8):
            if CORNER == 'fs':
                VRQ = VRQ_FS * epoch / int(EPOCHS*0.8)
            elif CORNER == 'sf':
                VRQ = VRQ_SF * epoch / int(EPOCHS*0.8)
            elif CORNER == 'ff':
                VRQ = VRQ_FF * epoch / int(EPOCHS*0.8)
            elif CORNER == 'ss':
                VRQ = VRQ_SS * epoch / int(EPOCHS*0.8)
            else:
                VRQ = VRQ_TT * epoch / int(EPOCHS*0.8)
        else:
            if CORNER == 'fs':
                VRQ = VRQ_FS
            elif CORNER == 'sf':
                VRQ = VRQ_SF
            elif CORNER == 'ff':
                VRQ = VRQ_FF
            elif CORNER == 'ss':
                VRQ = VRQ_SS
            else:
                VRQ = VRQ_TT
        print("VRQ: {}".format(VRQ))

        train(model, device, train_loader, optimizer, epoch)
        test_acc = test(model, device, test_loader, epoch)

        if save_model != None and (test_acc > acc) and epoch >= int(EPOCHS*0.8):
            acc = test_acc
            torch.save(model.state_dict(), save_model)
            print("Model saved!")

        scheduler.step()
        print('Time: {:.2f}'.format(time.time() - epoch_tstart))  
else:
    print('Evaluate the inference accuracy...')
    test(model, device, test_loader, 0)

tEnd = time.time()
print("Total simulation time: {:f} sec".format(tEnd - tStart))

