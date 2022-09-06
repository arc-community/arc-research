from typing import List, Set, Tuple
import math
import random
import uuid
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

import wandb

from arc.interface import Riddle
from arc.utils.dataset import load_riddle_from_file

from vit import ViT_NoEmbed
from warmup_scheduler import GradualWarmupScheduler


class RiddleLoader:
    colors_tables = {
        "official":
        # color values from fchollet's ARC repository https://github.com/fchollet/ARC/blob/b69465a8d1e628909ac58785583e9f41eda2cc51/apps/css/common.css#L14-L43
        [
            0x0,
            0x0074D9,
            0xFF4136,
            0x2ECC40,
            0xFFDC00,
            0xAAAAAA,
            0xF012BE,
            0xFF851B,
            0x7FDBFF,
            0x870C25,
        ],
        "mokole":
        # palette generated with https://mokole.com/palette.html
        [
            0x0,  # black
            0x006400,  # darkgreen
            0xFF0000,  # red
            0xFFD700,  # gold
            0x00FF00,  # lime
            0xE9967A,  # darksalmon
            0x00FFFF,  # aqua
            0x0000FF,  # blue
            0x6495ED,  # cornflower
            0xFF1493,  # deeppink
        ],
    }

    def __init__(self, file_names: List[Path], colors_table_name: str):
        self.file_names = file_names
        if len(self.file_names) == 0:
            raise RuntimeError("No riddles found.")
        self.order = list(range(len(self.file_names)))
        self.shuffle()

        if not colors_table_name in RiddleLoader.colors_tables:
            raise RuntimeError(f"Unspported color map '{colors_table_name}' specified.")

        color_table = RiddleLoader.colors_tables[colors_table_name]

        color_mapping = nn.Embedding(10, 3)
        for i, hex_color in enumerate(color_table):
            r = hex_color & 0xFF
            g = (hex_color >> 8) & 0xFF
            b = (hex_color >> 16) & 0xFF

            color_mapping.weight.data[i, 0] = r / 255.0
            color_mapping.weight.data[i, 1] = g / 255.0
            color_mapping.weight.data[i, 2] = b / 255.0
        self.color_mapping = color_mapping

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

    def load_batch(
        self, batch_size: int, device: torch.DeviceObjType
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        riddles = [self.next_riddle() for i in range(batch_size)]
        max_train_examples = max(len(r.train) for r in riddles)

        inputs = torch.zeros(batch_size, max_train_examples * 2 + 1, 3, 10, 10)  # train input+output = 2 + 1 test input
        targets = torch.zeros(batch_size, 3, 10, 10)
        target_boards = torch.zeros(batch_size, 10, 10, dtype=torch.long)
        color_mapping = self.color_mapping

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
            target_boards[i] = test_output_board
            targets[i] = color_mapping(test_output_board).permute(2, 0, 1)

        inputs, targets, target_boards = (x.to(device) for x in (inputs, targets, target_boards))
        return inputs, targets, target_boards


def parse_args() -> argparse.Namespace:
    # parse bool args correctly, see https://stackoverflow.com/a/43357954
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

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

    parser.add_argument("--eval_interval", default=500, type=int)
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

    parser.add_argument("--color_table", default="mokole", type=str)

    parser.add_argument("--save_eval_images", default=True, type=str2bool)

    return parser.parse_args()


@torch.no_grad()
def grad_norm(model_params):
    sqsum = 0.0
    for p in model_params:
        sqsum += (p.grad**2).sum().item()
    return math.sqrt(sqsum)


@torch.no_grad()
def quantize_output(y: torch.Tensor, color_mapping: nn.Embedding) -> torch.Tensor:
    batch_size = y.shape[0]
    y_flat = y.view(batch_size, 3, 10 * 10).permute(0, 2, 1)

    # calculate distances
    embedding = color_mapping.weight.to(y.device)
    distances_squared = (y_flat.unsqueeze(-1) - embedding.t().unsqueeze(0)).pow(2).sum(dim=-2)

    # encoding
    indices = torch.argmin(distances_squared, dim=-1).view(batch_size, 10, 10)
    return indices


@torch.no_grad()
def eval_model(
    device: torch.DeviceObjType,
    batch_size: int,
    num_batches: int,
    model,
    riddle_loader: RiddleLoader,
    loss_fn,
) -> Tuple[float, float]:
    model.eval()

    total_loss = 0
    total_accuracy = 0
    for i in range(num_batches):
        inputs, targets, target_boards = riddle_loader.load_batch(batch_size, device)
        b, n, *_ = inputs.shape
        inputs = inputs.view(b, n, -1)
        targets = targets.view(b, -1)

        y = model(inputs)

        loss = loss_fn(y, targets)
        total_loss += loss.item()

        t = quantize_output(y, riddle_loader.color_mapping)
        accuracy = (t == target_boards).sum() / t.numel()
        total_accuracy += accuracy.item()

    return total_loss / num_batches, total_accuracy / num_batches


@torch.no_grad()
def eval_model_visual(
    device: torch.DeviceObjType, batch_size: int, model, riddle_loader: RiddleLoader
) -> Tuple[float, float]:
    model.eval()

    inputs, targets, target_boards = riddle_loader.load_batch(batch_size, device)

    b, n, c, h, w = inputs.shape
    y = model(inputs.view(b, n, -1))

    # create error view
    error_view = torch.zeros(b, c, h, w, device=device)
    t = quantize_output(y, riddle_loader.color_mapping)
    error_view[:, 1] = (t == target_boards).float()  # correct pixels = green
    error_view[:, 0] = (t != target_boards).float()  # incorrect pixels = red

    # combine: examples, ground-truth, model ouput, error view
    combined = torch.cat(
        (
            inputs,
            targets.unsqueeze(1),
            y.view(targets.shape).unsqueeze(1),
            error_view.unsqueeze(1),
        ),
        dim=1,
    )
    num_per_riddle = combined.shape[1]

    riddle_grid = torchvision.utils.make_grid(
        combined.view(-1, 3, 10, 10), nrow=num_per_riddle, padding=2, normalize=False, value_range=(0, 1)
    )

    return riddle_grid


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

    # go over dataset and select all 10x10 riddles with equal input & output size and 3 training examples
    training_set = []

    print("filtering 10x10 riddles")
    max_trainig_examples = -1  # 128  # for dev only
    for fn in tqdm(file_names):
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

    rl = RiddleLoader(file_names=train_file_names, colors_table_name=args.color_table)
    riddle_loader_eval = RiddleLoader(file_names=eval_file_names, colors_table_name=args.color_table)
    batch, targets, _ = rl.load_batch(1, device)

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
            eval_loss, eval_accuracy = eval_model(
                device,
                batch_size,
                args.num_eval_batches,
                model,
                riddle_loader_eval,
                loss_fn,
            )
            print(f"step: {step}; eval loss: {eval_loss:.4e}; eval accuracy: {eval_accuracy:.2%};")

            eval_grid = eval_model_visual(device, 32, model, riddle_loader_eval)
            if args.save_eval_images:
                torchvision.utils.save_image(eval_grid, f"{experiment_name}_eval_{step:08d}.png")
            eval_image = wandb.Image(eval_grid, caption="input, GT, prediction, error")

            wandb.log({"eval.loss": eval_loss, "eval.accuracy": eval_accuracy, "eval.image": eval_image}, step=step)

        model.train()
        optimizer.zero_grad()

        batch, targets, target_boards = rl.load_batch(batch_size, device)
        b, n, *_ = batch.shape
        batch = batch.view(b, n, -1)
        target = targets.view(b, -1)

        y = model(batch)
        loss = loss_fn(y, target)

        loss.backward()
        gn = grad_norm(model.parameters())
        optimizer.step()
        lr_scheduler.step()

        t = quantize_output(y, rl.color_mapping)
        train_accuracy = (t == target_boards).sum() / t.numel()

        if step % 10 == 0:
            print(
                f"step: {step}; train loss: {loss.item():.4e}; train accuracy: {train_accuracy.item():.2%} lr: {lr_scheduler.get_last_lr()[0]:.3e}; grad_norm: {gn:.3e}"
            )

        wandb.log(
            {
                "train.loss": loss.item(),
                "train.accuracy": train_accuracy.item(),
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
