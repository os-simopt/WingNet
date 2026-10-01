# wingnet/model/architecture.py

import torch
from torch import nn
import config


IMG_SIZE = config.CONFIG["model"]["image_size"]
DROPOUT = config.CONFIG["model"]["dropout"]
UCONV_DEPTH = config.CONFIG["model"]["uconv_depth"]


class SqueezeExcite(nn.Module):
    def __init__(self, n: int, reduction: int = 16):
        super(SqueezeExcite, self).__init__()

        self.seq = torch.nn.Sequential(
            nn.Linear(n, n // reduction),
            nn.PReLU(n // reduction),
            nn.Linear(n // reduction, n),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.mean(dim=(2, 3))
        out = self.seq(out)
        return x * out.view(x.shape[0], x.shape[1], 1, 1)


class ResConvSE(nn.Module):
    def __init__(self, n: int, scale: int):
        super(ResConvSE, self).__init__()
        self.scale = scale

        self.pr1 = nn.PReLU(n)
        self.ln1 = nn.Conv2d(in_channels=n, out_channels=n * scale, kernel_size=3, padding=1)
        self.pr2 = nn.PReLU(n * scale)
        self.ln2 = nn.Conv2d(in_channels=n * scale, out_channels=n * scale, kernel_size=3, padding=1)

        self.se = SqueezeExcite(n * scale, 8)

        # Dropout-Raten unverändert, jetzt aus config
        self.do1 = nn.Dropout2d(p=DROPOUT["resconv_do1"], inplace=True)
        self.do2 = nn.Dropout2d(p=DROPOUT["resconv_do2"], inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pr1(x)
        self.do1(out)
        out = self.ln1(out)
        out = self.pr2(out)
        self.do2(out)
        out = self.ln2(out)

        out = self.se.forward(out)

        out += x.repeat(1, self.scale, 1, 1)

        return out


class UConvImpl(nn.Module):
    def __init__(self, n: int, mid_module: nn.Module | None = None):
        super(UConvImpl, self).__init__()
        if n <= 0:
            raise Exception("invalid params")

        self.down = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels=n, out_channels=n * 2, kernel_size=3, padding=1),
                    nn.PReLU(n * 2),
                ),
                nn.MaxPool2d(2),
            ]
        )

        self.up = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels=n * 4, out_channels=n, kernel_size=3, padding=1),
                    nn.PReLU(n),
                ),
                nn.ConvTranspose2d(in_channels=n * 2, out_channels=n * 2, kernel_size=2, stride=2),
            ]
        )

        if mid_module is None:
            self.rn1 = nn.Sequential(
                nn.Conv2d(in_channels=n * 2, out_channels=n * 2, kernel_size=3, padding=1),
                nn.PReLU(n * 2),
                SqueezeExcite(n * 2),
            )
        else:
            self.rn1 = mid_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.down[0](x)
        skip = out
        out = self.down[1](out)

        out = self.rn1(out)

        out = self.up[1].forward(out)
        out = self.up[0].forward(torch.cat((skip, out), dim=1))

        return out


def recursive_build_u_conv(n: int, layers: int) -> nn.Module:
    if layers <= 0:
        raise Exception("invalid params")

    layers -= 1
    n *= 2**layers

    mod = UConvImpl(n)
    for _ in range(layers):
        n //= 2
        mod = UConvImpl(n, mod)

    return mod


class UConvDigest(nn.Module):
    def __init__(self, n: int, multiplier: int, depth: int = UCONV_DEPTH):
        super(UConvDigest, self).__init__()

        if n <= 0 or depth <= 1 or multiplier <= 1:
            raise Exception("invalid params")

        self.factor = multiplier + 1
        self.multiplier = multiplier

        nf = n * self.factor
        self.n = n
        self.depth = depth

        self.pr1 = nn.PReLU(nf)

        # Dropout-Raten unverändert, aus config
        self.do1 = nn.Dropout2d(p=DROPOUT["uconv_do1"], inplace=True)
        self.do2 = nn.Dropout2d(p=DROPOUT["uconv_do2"], inplace=True)

        self.down = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels=nf, out_channels=nf * 2, kernel_size=3, padding=1),
                    nn.PReLU(nf * 2),
                ),
                nn.MaxPool2d(2),
            ]
        )

        self.up = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=nf * 4, out_channels=multiplier * n, kernel_size=3, padding=1
                    ),
                    nn.PReLU(multiplier * n),
                ),
                nn.ConvTranspose2d(in_channels=nf * 2, out_channels=nf * 2, kernel_size=2, stride=2),
            ]
        )

        self.rn1 = recursive_build_u_conv(nf * 2, depth - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pr1(x)
        self.do1(out)

        out = self.down[0](out)
        skip = out
        out = self.down[1](out)

        out = self.rn1.forward(out)

        out = self.up[1].forward(out)
        out = torch.cat((skip, out), dim=1)
        self.do2(out)
        out = self.up[0].forward(out)

        sl1 = x[:, 0 : self.n, :, :]
        sl2 = x[:, self.n : self.n * self.factor, :, :]

        out = out + sl1.repeat(1, self.multiplier, 1, 1)
        out = out + sl2

        return out


class WingNet(nn.Module):
    def __init__(self, digest_channels: int, in_channels: int = 3, out_shape: int = 4):
        super(WingNet, self).__init__()
        self.image_size = IMG_SIZE

        if digest_channels % in_channels:
            raise Exception("digest must be multiple of in_channels")

        self.digest_net = nn.Sequential(
            UConvDigest(in_channels, digest_channels // in_channels, UCONV_DEPTH),
            nn.Tanh(),
        )

        self.init = nn.Parameter(torch.randn(1, digest_channels, 1, 1))

        self.final_classifier = nn.Sequential(
            ResConvSE(digest_channels, 2),
            nn.MaxPool2d(kernel_size=4, stride=4),
            ResConvSE(digest_channels * 2, 2),
            nn.MaxPool2d(kernel_size=4, stride=4),
            ResConvSE(digest_channels * 4, 2),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Linear(digest_channels * 32, out_shape),
            nn.Softmax(1),
        )

    @torch.jit.export
    def init_digest(self, n: int) -> torch.Tensor:
        return self.init.clone().repeat(n, 1, self.image_size, self.image_size)

    @torch.jit.export
    def tick(self, feed: torch.Tensor, digest: torch.Tensor) -> torch.Tensor:
        # single recurrent step without eval of classifier part
        combined = torch.cat((feed, digest), dim=1)
        return self.digest_net(combined)

    @torch.jit.export
    def finalize(self, digest: torch.Tensor) -> torch.Tensor:
        # simply calls classifier
        return self.final_classifier(digest)

    @torch.jit.export
    def fast_tick(self, feed_digest: torch.Tensor) -> torch.Tensor:
        # performs tick but assumes input is already a concatenation
        return self.digest_net(feed_digest)

    @torch.jit.export
    def get_init(self) -> torch.Tensor:
        return self.init

    @torch.jit.export
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # evals sequence and gets prediction at each index
        d = self.init_digest(1)
        ret_list = []
        for feed in x:
            d = self.tick(feed.unsqueeze(0), d)
            ret_list.append(self.finalize(d))
        return torch.cat(ret_list)

    @torch.jit.export
    def simple_forward(self, x: torch.Tensor) -> torch.Tensor:
        # evals sequence and only predicts at the end
        d = self.init_digest(1)
        for feed in x:
            d = self.tick(feed.unsqueeze(0), d)
        return self.finalize(d)
