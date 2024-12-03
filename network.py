import torch.nn as nn
from torch.nn.functional import normalize
import torch


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )


    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    def __init__(self, view, args,input_size, feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
        )
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, class_num),
            nn.Softmax(dim=1)
        )

        self.view = view

        self.feature_up_module = nn.Sequential(
            nn.Linear(high_feature_dim, feature_dim),
        )

        latent_dim = args.hidden_dim // 2
        self.act = nn.ReLU(inplace=True)

        self.map_layer = nn.Sequential(
            nn.Linear(args.hidden_dim * args.views, args.hidden_dim, bias=True),
            nn.BatchNorm1d(args.hidden_dim),
            self.act,
        )
        self.fusion = nn.Sequential(
            nn.Linear(args.hidden_dim, latent_dim, bias=False),
            nn.BatchNorm1d(latent_dim),
            self.act,
            self._make_layers(latent_dim, latent_dim, self.act, args.mlp_layers, args.use_bn),
            nn.Linear(latent_dim, args.hidden_dim, bias=False),
        )


    def forward_label(self, z):
        feature=self.feature_up_module(z)
        qs = self.label_contrastive_module(feature)
        return qs

    def forwardZ(self, h):

        h = torch.cat(h, dim=1)
        z = self.map_layer(h)
        res = z
        z = self.fusion(z)
        z += res
        return z

    def _make_layers(self, in_features, out_features, act, num_layers, bn=False):
        layers = nn.ModuleList()
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, out_features, bias=False))
            if bn:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(act)
        return nn.Sequential(*layers)

    def forward(self, xs):
        hs = []
        qs = []
        xrs = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.feature_contrastive_module(z), dim=1)
            q = self.label_contrastive_module(z)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            qs.append(q)
            xrs.append(xr)
        commonZ=self.forwardZ(hs)
        label=self.forward_label(commonZ)
        return hs, qs, xrs, zs,commonZ,label

    def forward_cluster(self, xs):
        qs = []
        preds = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            q = self.label_contrastive_module(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        return qs, preds
