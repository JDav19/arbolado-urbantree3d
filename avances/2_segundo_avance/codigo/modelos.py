"""PointNet, PointNet++ (SSG) y DGCNN para clasificacion de un arbol completo."""
import torch
import torch.nn as nn


class PointNetCls(nn.Module):
    def __init__(self, n_clases=6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1, bias=False), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1, bias=False), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, n_clases),
        )

    def forward(self, x):  # (B,3,N)
        return self.fc(self.mlp(x).max(dim=2)[0])


# ---------- PointNet++ (agrupamiento de una sola escala) ----------
def dist2(a, b):  # a (B,M,3), b (B,N,3)
    return (a ** 2).sum(-1, keepdim=True) - 2 * a @ b.transpose(1, 2) + (b ** 2).sum(-1).unsqueeze(1)


def muestreo_fps(xyz, m):
    B, N, _ = xyz.shape
    idx = torch.zeros(B, m, dtype=torch.long)
    dist = torch.full((B, N), 1e10)
    lejano = torch.zeros(B, dtype=torch.long)
    ar = torch.arange(B)
    for i in range(m):
        idx[:, i] = lejano
        c = xyz[ar, lejano].unsqueeze(1)
        d = ((xyz - c) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        lejano = dist.argmax(dim=1)
    return idx


def tomar(x, idx):  # x (B,N,C), idx (B,...)
    B = x.shape[0]
    forma = idx.shape
    ar = torch.arange(B).view(B, *([1] * (len(forma) - 1))).expand(*forma)
    return x[ar, idx]


class SetAbstraction(nn.Module):
    def __init__(self, n_centros, radio, k, canales):
        super().__init__()
        self.n_centros, self.radio, self.k = n_centros, radio, k
        capas, previo = [], canales[0]
        for c in canales[1:]:
            capas += [nn.Conv2d(previo, c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU()]
            previo = c
        self.mlp = nn.Sequential(*capas)

    def forward(self, xyz, feat):  # xyz (B,N,3), feat (B,N,C) o None
        idx = muestreo_fps(xyz, self.n_centros)
        nuevo = tomar(xyz, idx)
        d = dist2(nuevo, xyz)
        d_k, vec = d.topk(self.k, dim=-1, largest=False)
        fuera = d_k > self.radio ** 2
        vec = torch.where(fuera, vec[..., :1].expand_as(vec), vec)
        rel = tomar(xyz, vec) - nuevo.unsqueeze(2)
        g = rel if feat is None else torch.cat([rel, tomar(feat, vec)], dim=-1)
        g = self.mlp(g.permute(0, 3, 1, 2)).max(dim=-1)[0]  # (B,C,M)
        return nuevo, g.transpose(1, 2)


class PointNetPP(nn.Module):
    def __init__(self, n_clases=6, n_puntos=1024):
        super().__init__()
        self.sa1 = SetAbstraction(256, 0.2, 32, [3, 64, 64, 128])
        self.sa2 = SetAbstraction(64, 0.4, 32, [128 + 3, 128, 128, 256])
        self.glob = nn.Sequential(
            nn.Conv1d(256 + 3, 256, 1, bias=False), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 512, 1, bias=False), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512, bias=False), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, n_clases),
        )

    def forward(self, x):  # (B,3,N)
        xyz = x.transpose(1, 2).contiguous()
        x1, f1 = self.sa1(xyz, None)
        x2, f2 = self.sa2(x1, f1)
        g = self.glob(torch.cat([x2, f2], dim=-1).transpose(1, 2)).max(dim=2)[0]
        return self.fc(g)


# ---------- DGCNN ----------
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = (x ** 2).sum(dim=1, keepdim=True)
    return (-xx - inner - xx.transpose(2, 1)).topk(k=k, dim=-1)[1]


def graph_feature(x, k):
    B, C, N = x.shape
    idx = knn(x, k)
    base = torch.arange(B).view(-1, 1, 1) * N
    idx = (idx + base).view(-1)
    xt = x.transpose(2, 1).contiguous()
    vec = xt.view(B * N, -1)[idx].view(B, N, k, C)
    centro = xt.view(B, N, 1, C).expand(-1, -1, k, -1)
    return torch.cat([vec - centro, centro], dim=3).permute(0, 3, 1, 2).contiguous()


class DGCNNCls(nn.Module):
    def __init__(self, n_clases=6, k=16):
        super().__init__()
        self.k = k
        def bloque(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.LeakyReLU(0.2))
        self.e1, self.e2, self.e3 = bloque(6, 64), bloque(128, 64), bloque(128, 128)
        self.emb = nn.Sequential(nn.Conv1d(256, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        self.fc = nn.Sequential(
            nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512), nn.LeakyReLU(0.2), nn.Dropout(0.4),
            nn.Linear(512, 256, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2), nn.Dropout(0.4),
            nn.Linear(256, n_clases),
        )

    def forward(self, x):
        x1 = self.e1(graph_feature(x, self.k)).max(dim=-1)[0]
        x2 = self.e2(graph_feature(x1, self.k)).max(dim=-1)[0]
        x3 = self.e3(graph_feature(x2, self.k)).max(dim=-1)[0]
        e = self.emb(torch.cat([x1, x2, x3], dim=1))
        return self.fc(torch.cat([e.max(dim=2)[0], e.mean(dim=2)], dim=1))


def crear(nombre):
    return {"PointNet": PointNetCls, "PointNet++": PointNetPP, "DGCNN": DGCNNCls}[nombre]()
