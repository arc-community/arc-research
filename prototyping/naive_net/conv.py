from operator import mod
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from arc.interface import Riddle
from arc.utils import dataset


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=bias,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class Residual(nn.Module):
    def __init__(
        self,
        in_planes=64,
        hidden_planes=128,
        normalize=nn.BatchNorm2d,
        nonlinearity=nn.LeakyReLU,
    ):
        super(Residual, self).__init__()
        self.activation_fn = nonlinearity()
        self.block = nn.Sequential(
            conv3x3(in_planes=in_planes, out_planes=hidden_planes),
            normalize(hidden_planes),
            nonlinearity(inplace=True),
            conv1x1(in_planes=hidden_planes, out_planes=in_planes),
            normalize(in_planes),
        )

    def forward(self, x):
        x = self.block(x) + x
        return self.activation_fn(x)


class ResidualStack(nn.Module):
    def __init__(self, num_layers=3, latent_dim=64, hidden_dim=128):
        super(ResidualStack, self).__init__()
        self.num_layers = num_layers

        layers = []
        for _ in range(num_layers):
            layers.append(Residual(latent_dim, hidden_dim))
        self._stack = nn.Sequential(*layers)

    def forward(self, x):
        return self._stack(x)


class ConvNet(nn.Module):
    def __init__(
        self, in_planes=10, out_planes=10, latent_dim=64, num_layers=3, hidden_dim=128
    ):
        super(ConvNet, self).__init__()
        self.conv0 = conv3x3(in_planes=in_planes, out_planes=latent_dim)
        self.residual_stack = ResidualStack(
            num_layers=num_layers, latent_dim=latent_dim, hidden_dim=hidden_dim
        )
        self.output_projection = conv1x1(in_planes=latent_dim, out_planes=out_planes)

    def forward(self, x):
        x = self.conv0(x)
        x = F.leaky_relu(x, inplace=True)
        x = self.residual_stack(x)
        x = self.output_projection(x)
        return x


def conv_solve(
    riddle: Riddle,
    max_steps=100,
    lr=1e-4,
    weight_decay=0.1,
    num_layers=2,
    latent_dim=64,
    hidden_dim=128,
):
    train_pairs = riddle.train

    inputs = [torch.from_numpy(bp.input.as_np) for bp in train_pairs]
    targets = [torch.from_numpy(bp.output.as_np) for bp in train_pairs]

    model = ConvNet(
        in_planes=10,
        out_planes=10,
        latent_dim=latent_dim,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
        amsgrad=False,
    )
    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    model.train()
    for step in range(max_steps):
        optimizer.zero_grad()

        for i in range(len(inputs)):
            x = inputs[i].unsqueeze(0)
            target = targets[i]
            target = target.view(1, target.shape[0] * target.shape[1])

            # embed input
            x = F.one_hot(x, num_classes=10).permute(0, 3, 1, 2).float()

            y = model(x)
            y = y.view(1, 10, target.shape[1])
            loss = loss_fn(y, target)

            loss.backward()

        optimizer.step()

    # eval on test
    test_pairs = riddle.test
    inputs = [torch.from_numpy(bp.input.as_np) for bp in test_pairs]
    targets = [torch.from_numpy(bp.output.as_np) for bp in test_pairs]

    with torch.no_grad():
        model.eval()
        x = inputs[0].unsqueeze(0)
        target = targets[0].unsqueeze(0)

        # embed input
        x = F.one_hot(x, num_classes=10).permute(0, 3, 1, 2).float()
        y = model.forward(x)

        y = torch.argmax(y, dim=1, keepdim=False)
        correct = torch.count_nonzero(y == target)
        # print(riddle.riddle_id, correct / y.numel(), torch.all(y == target))
        return torch.all(y == target), y


def main():

    hparam_configs = [
        dict(
            max_steps=250,
            lr=1e-4,
            weight_decay=0.1,
            num_layers=1,
            latent_dim=32,
            hidden_dim=64,
        ),
        dict(
            max_steps=250,
            lr=1e-4,
            weight_decay=0.1,
            num_layers=2,
            latent_dim=32,
            hidden_dim=64,
        ),
        dict(
            max_steps=500,
            lr=1e-4,
            weight_decay=0.1,
            num_layers=1,
            latent_dim=64,
            hidden_dim=128,
        ),
        dict(
            max_steps=100,
            lr=1e-4,
            weight_decay=0.1,
            num_layers=2,
            latent_dim=64,
            hidden_dim=128,
        ),
        dict(
            max_steps=500,
            lr=1e-4,
            weight_decay=0.1,
            num_layers=2,
            latent_dim=64,
            hidden_dim=128,
        ),
        dict(
            max_steps=1000,
            lr=1e-4,
            weight_decay=0.1,
            num_layers=3,
            latent_dim=64,
            hidden_dim=128,
        ),
        dict(
            max_steps=2000,
            lr=1e-4,
            weight_decay=0.1,
            num_layers=3,
            latent_dim=64,
            hidden_dim=128,
        ),
        dict(
            max_steps=2000,
            lr=1e-4,
            weight_decay=0.1,
            num_layers=5,
            latent_dim=256,
            hidden_dim=512,
        ),
    ]

    train_riddle_ids = dataset.get_riddle_ids(["training"])
    eval_riddle_ids = dataset.get_riddle_ids(["evaluation"])

    def run_solve(riddle_ids, hparams):
        solved = []
        for id in riddle_ids:
            riddle = dataset.load_riddle_from_id(id)

            # only try riddles with equal input/output size
            if all(
                p.input.num_rows == p.output.num_rows
                and p.input.num_cols == p.output.num_cols
                for p in riddle.train
            ) and all(
                p.input.num_rows == p.output.num_rows
                and p.input.num_cols == p.output.num_cols
                for p in riddle.test
            ):
                success, result = conv_solve(riddle, **hparams)
                if success:
                    solved.append(riddle.riddle_id)
        return solved

    for i, hparams in enumerate(hparam_configs):
        print(f"hparams #{i}: ", hparams)

        train_solved = run_solve(train_riddle_ids, hparams)
        eval_solved = run_solve(eval_riddle_ids, hparams)

        print(
            f"Train: {len(train_solved)}/{len(train_riddle_ids)}, ({len(train_solved)/len(train_riddle_ids):%})"
        )
        print(
            f"Eval: {len(eval_solved)}/{len(eval_riddle_ids)}, ({len(eval_solved)/len(eval_riddle_ids):%})"
        )

        solved_riddles = len(train_solved) + len(eval_solved)
        total_riddles = len(eval_riddle_ids) + len(train_riddle_ids)
        print(
            f"Combined: {solved_riddles/total_riddles}, ({solved_riddles/total_riddles:%})"
        )

        print("Correctly predicted train:", train_solved)
        print("Correctly predicted eval:", eval_solved)
        print()


if __name__ == "__main__":
    main()
