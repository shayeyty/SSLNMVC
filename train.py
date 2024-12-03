from loss import Loss
from network import Network
from metric import valid
from torch.utils.data import Dataset
import argparse
from dataloader import load_data
import os
import torch.nn.functional as F
import torch


# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
# NGs
# BDGP
# synthetic3d
# Fashion
# NoisyMNIST

Dataname = 'NoisyMNIST'

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--feature_dim", default=1200)
parser.add_argument("--high_feature_dim", default=200)
parser.add_argument("--hidden_dim", default=200)
parser.add_argument('--activation', type=str, default='relu', help='Activation function.')
parser.add_argument('--views', type=int, default=2, help='Number of views.')
parser.add_argument('--use_bn', type=bool, default=False, help='Whether to use batch normalization.')
parser.add_argument('--mlp_layers', type=int, default=1, help='Number of MLP layers.')
parser.add_argument("--alpha", default=1, help='Contrast loss hyperparameters.')
parser.add_argument("--beta", default=0.05, help='Label alignment loss hyperparameters.')


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == "Caltech-2V":
    args.con_epochs = 50
    args.views = 2
    seed = 13
if args.dataset == "Caltech-3V":
    args.con_epochs = 100
    args.views = 3
    seed = 10
if args.dataset == "Caltech-4V":
    args.con_epochs = 50
    args.views = 4
    seed = 5
if args.dataset == "Caltech-5V":
    args.con_epochs = 60
    args.views = 5
    seed = 5
if args.dataset == "NGs":
    args.con_epochs = 120
    args.views = 3
    seed = 100
if args.dataset == "BDGP":
    args.con_epochs = 10
    args.views = 2
    seed = 13
if args.dataset == "synthetic3d":
    args.con_epochs = 80
    args.views = 3
    seed = 10
if args.dataset == "Fashion":
    args.con_epochs = 80
    args.views = 3
    seed = 10
if args.dataset == "NoisyMNIST":
    args.con_epochs = 150
    # args.con_epochs = 200
    args.views = 2
    seed = 13


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, xrs, _ ,_,_= model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


def contrastive_train(epoch):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        hs, qs, xrs, zs, commonZ, label= model(xs)
        loss_list = []
        for v in range(view):
            for w in range(v+1, view):
                loss_list.append(criterion.forward_feature(hs[v], hs[w])*args.alpha)
                loss_list.append(criterion.forward_label(qs[v], qs[w])*args.alpha)
            loss_list.append(F.kl_div(F.log_softmax(label, dim=1), F.softmax(qs[v], dim=1), reduction='sum')*args.beta)
            loss_list.append(mes(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))
    return tot_loss


accs = []
nmis = []
purs = []
if not os.path.exists('./models'):
    os.makedirs('./models')
T = 1


for i in range(T):
    print("ROUND:{}".format(i+1))
    setup_seed(seed)
    model = Network(view,args, dims, args.feature_dim, args.high_feature_dim, class_num, device)
    print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)

    epoch = 1
    while epoch <= args.mse_epochs:
        pretrain(epoch)
        epoch += 1
    accuracies=[]
    nmis = []
    purs = []
    losses = []
    while epoch <= args.mse_epochs + args.con_epochs:
        tot_loss=contrastive_train(epoch)
        if epoch == args.mse_epochs + args.con_epochs:
            acc, nmi, pur = valid(model,args, device, dataset, view, data_size)
            state = model.state_dict()
            torch.save(state, './models/' + args.dataset + '.pth')
            print('Saving..')
            accs.append(acc)
            nmis.append(nmi)
            purs.append(pur)
        epoch += 1


