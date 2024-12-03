import torch
from network import Network
from metric import valid
import argparse
from dataloader import load_data

# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
# NGs
# BDGP
# synthetic3d
# NoisyMNIST
# Fashion

Dataname = 'Caltech-2V'

parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
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


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if args.dataset == "Caltech-2V":
    args.views = 2
if args.dataset == "Caltech-3V":
    args.views = 3
if args.dataset == "Caltech-4V":
    args.views = 4
if args.dataset == "Caltech-5V":
    args.views = 5
if args.dataset == "NGs":
    args.views = 3
if args.dataset == "NoisyMNIST":
    args.views = 2
if args.dataset == "BDGP":
    args.views = 2
if args.dataset == "synthetic3d":
    args.views = 3
if args.dataset == "Fashion":
    args.views = 3
if args.dataset == "NoisyMNIST":
    args.views = 2


dataset, dims, view, data_size, class_num = load_data(args.dataset)
model = Network(view, args,dims, args.feature_dim, args.high_feature_dim, class_num, device)
model = model.to(device)
checkpoint = torch.load('./models/' + args.dataset + '.pth')
model.load_state_dict(checkpoint)
print("Dataset:{}".format(args.dataset))
print("Datasize:" + str(data_size))
print("Loading models...")
valid(model, args, device, dataset, view, data_size)
