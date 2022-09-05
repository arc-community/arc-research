import math
from turtle import color
from typing import List, Set, Tuple

import argparse
import random
import uuid

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from vit import ViT_NoEmbed

from pathlib import Path

from arc.interface import Riddle
from arc.utils.dataset import load_riddle_from_file

import wandb

from warmup_scheduler import GradualWarmupScheduler


class RiddleLoader:
    def __init__(self, file_names: List[Path]):
        self.file_names = file_names
        if len(self.file_names) == 0:
            raise RuntimeError("No riddles found.")
        self.order = list(range(len(self.file_names)))
        self.shuffle()

    def shuffle(self) -> None:
        random.shuffle(self.order)
        self.next_index = 0

    def next_riddle(self) -> Tuple[Riddle, Set[str]]:
        if self.next_index >= len(self.file_names):
            self.shuffle()
        fn = self.file_names[self.order[self.next_index]]
        self.next_index += 1
        r = load_riddle_from_file(fn)
        return r

    def load_batch(self, batch_size: int, device: torch.DeviceObjType) -> Tuple:
        riddles = [self.next_riddle() for i in range(batch_size)]
        max_train_examples = max(len(r.train) for r in riddles)

        # color valuess taken from ARC game https://volotat.github.io/ARC-Game/
        # color_table = [
        #     0x0,
        #     0x0074D9,
        #     0xFF4136,
        #     0x2ECC40,
        #     0xFFDC00,
        #     0xAAAAAA,
        #     0xF012BE,
        #     0xFF851B,
        #     0x7FDBFF,
        #     0x870C25,
        # ]

        # # palette generated with https://mokole.com/palette.html
        color_table = [
            0x0, # black
            0x006400, # darkgreen
            0xff0000, # red
            0xffd700, # gold
            0x00ff00, # lime
            0xe9967a, # darksalmon
            0x00ffff, # aqua
            0x0000ff, # blue
            0x6495ed, # cornflower
            0xff1493, # deeppink
        ]

        color_mapping = nn.Embedding(10, 3)
        for i, hex_color in enumerate(color_table):
            r = hex_color & 0xFF
            g = (hex_color >> 8) & 0xFF
            b = (hex_color >> 16) & 0xFF

            color_mapping.weight.data[i, 0] = r / 255.0
            color_mapping.weight.data[i, 1] = g / 255.0
            color_mapping.weight.data[i, 2] = b / 255.0

        inputs = torch.zeros(batch_size, max_train_examples * 2 + 1, 3, 10, 10)  # train input+output = 2 + 1 test input
        targets = torch.zeros(batch_size, 3, 10, 10)

        # convert board to tensor and map colors
        for i, r in enumerate(riddles):

            all_pairs = list(r.train)
            all_pairs.extend(r.test)
            random.shuffle(all_pairs)

            train_pairs = all_pairs[:-1]
            test_pair = all_pairs[-1]

            # encode train examples
            for j, t in enumerate(train_pairs):
                # add input board
                input_board = torch.from_numpy(t.input.np)
                inputs[i, j * 2] = color_mapping(input_board).permute(2, 0, 1)
                # add output board
                output_board = torch.from_numpy(t.output.np)
                inputs[i, j * 2 + 1] = color_mapping(output_board).permute(2, 0, 1)
            
            # add test input
            test_input_board = torch.from_numpy(test_pair.input.np)
            inputs[i, len(r.train) * 2] = color_mapping(test_input_board).permute(2, 0, 1)

            test_output_board = torch.from_numpy(test_pair.output.np)
            targets[i] = color_mapping(test_output_board).permute(2, 0, 1)

        inputs = inputs.to(device)
        targets = targets.to(device)
        return inputs, targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str, help="device to use")
    parser.add_argument("--device_index", default=0, type=int, help="device index")
    parser.add_argument(
        "--manual_seed",
        default=958137723,
        type=int,
        help="initialization of pseudo-RNG",
    )

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_steps", default=50000, type=int)
    parser.add_argument("--warmup", default=500, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--checkpoint_interval", default=25000, type=int)

    parser.add_argument("--dim", default=1024, type=int)
    parser.add_argument("--depth", default=24, type=int)
    parser.add_argument("--heads", default=16, type=int)
    parser.add_argument("--mlp_dim", default=4096, type=int)

    parser.add_argument("--eval_interval", default=1000, type=int)
    parser.add_argument("--num_eval_batches", default=32, type=int)

    train_riddle_folder = "/data/synth_riddles/unary/unary_depth_2_10k_10x10"
    # train_riddle_folder = "~/.arc/cache/dataset/evaluation"
    parser.add_argument("--riddle_folder", default=train_riddle_folder, type=str)

    parser.add_argument("--wandb", default=False, action="store_true")
    parser.add_argument("--project", default="arc_vit", type=str, help="project name for wandb")
    parser.add_argument(
        "--name",
        default="arc_vit_" + uuid.uuid4().hex,
        type=str,
        help="wandb experiment name",
    )

    return parser.parse_args()


@torch.no_grad()
def grad_norm(model_params):
    sqsum = 0.0
    for p in model_params:
        sqsum += (p.grad**2).sum().item()
    return math.sqrt(sqsum)


@torch.no_grad()
def eval_model(
    device: torch.DeviceObjType,
    batch_size: int,
    num_batches: int,
    model,
    riddle_loader: RiddleLoader,
    loss_fn,
) -> float:
    model.eval()

    total_loss = 0
    for i in range(num_batches):
        inputs, targets = riddle_loader.load_batch(batch_size, device)
        b, n, _, _, _ = inputs.shape
        inputs = inputs.view(b, n, -1)
        targets = targets.view(b, -1)

        y = model.forward(inputs)
        loss = loss_fn(y, targets)
        total_loss += loss.item()

    return total_loss / num_batches


def main():
    print(f"Using pytorch version {torch.__version__}")
    args = parse_args()

    print("Effective args:", args)

    experiment_name = args.name
    if args.wandb:
        wandb_mode = "online"
        wandb.login()
    else:
        wandb_mode = "disabled"
    wandb.init(project=args.project, config=vars(args), mode=wandb_mode, name=args.name)

    torch.manual_seed(args.manual_seed)
    device = torch.device(args.device, args.device_index)
    file_names = list(Path(args.riddle_folder).expanduser().glob("*.json"))

    print("filtering riddles")

    # max_trainig_examples = 128
    max_trainig_examples = -1

    # go over dataset and select all 10x10 riddles with equal input & output size and 3 training examples
    training_set = []
    for fn in file_names:
        r = load_riddle_from_file(fn)
        if all(
            bp.input.num_cols == 10
            and bp.input.num_rows == 10
            and bp.output.num_cols == 10
            and bp.output.num_rows == 10
            for bp in r.train
        ) and all(
            bp.input.num_cols == 10
            and bp.input.num_rows == 10
            and bp.output.num_cols == 10
            and bp.output.num_rows == 10
            for bp in r.test
        ):
            training_set.append(fn)
            if max_trainig_examples > 0 and len(training_set) > max_trainig_examples:
                break

    num_train_riddles = len(training_set) * 80 // 100
    eval_file_names = training_set[num_train_riddles:]
    train_file_names = training_set[:num_train_riddles]
    print(f"Num train riddles: {len(train_file_names)}")
    print(f"Num eval riddles: {len(eval_file_names)}")

    loss_fn = nn.MSELoss()

    rl = RiddleLoader(train_file_names)
    riddle_loader_eval = RiddleLoader(file_names=eval_file_names)
    batch, targets = rl.load_batch(1, device)

    num_patches = batch.shape[1]
    patch_dim = batch.shape[2] * batch.shape[3] * batch.shape[4]

    model = ViT_NoEmbed(
        num_patches=num_patches,
        patch_dim=patch_dim,
        num_classes=targets[0].numel(),
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
    )
    model.to(device)

    batch_size = args.batch_size
    max_steps = args.max_steps

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        amsgrad=False,
    )

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_steps)
    lr_scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1.0,
        total_epoch=args.warmup,
        after_scheduler=scheduler_cosine,
    )

    checkpoint_interval = args.checkpoint_interval
    for step in range(1, max_steps + 1):
        if step % args.eval_interval == 0:
            eval_loss = eval_model(
                device,
                batch_size,
                args.num_eval_batches,
                model,
                riddle_loader_eval,
                loss_fn,
            )
            print(
                f"step: {step}; eval loss: {eval_loss:.4e};"
            )
            wandb.log({"eval.loss": eval_loss}, step=step)

        model.train()
        optimizer.zero_grad()

        batch, targets = rl.load_batch(batch_size, device)
        b, n, _, _, _ = batch.shape
        batch = batch.view(b, n, -1)
        target = targets.view(b, -1)
        
        y = model.forward(batch)
        loss = loss_fn(y, target)

        loss.backward()
        gn = grad_norm(model.parameters())
        optimizer.step()
        lr_scheduler.step()

        if step % 10 == 0:
            print(
                f"step: {step}; train loss: {loss.item():.4e}; lr: {lr_scheduler.get_last_lr()[0]:.3e}; grad_norm: {gn:.3e}"
            )

        wandb.log(
            {
                "train.loss": loss.item(),
                "train.lr": lr_scheduler.get_last_lr()[0],
                "train.grad_norm": gn,
            },
            step=step,
        )

        if step % checkpoint_interval == 0:
            # write model_checkpoint
            fn = "{}_checkpoint_{:07d}.pth".format(experiment_name, step)
            print("writing file: " + fn)
            torch.save(
                {
                    "step": step,
                    "lr": lr_scheduler.get_last_lr(),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "warmup_scheduler_state_dict": lr_scheduler.state_dict(),
                    "cosine_scheduler_state_dict": scheduler_cosine.state_dict(),
                    "args": args,
                },
                fn,
            )



if __name__ == "__main__":
    main()
