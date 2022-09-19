# (formatted with black --line-length 120)
from typing import Dict, List, Set, Tuple
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

    def __init__(
        self,
        file_names: List[Path],
        colors_table_name: str,
        shuffle_train_test: bool = True,
    ):
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
        self.shuffle_train_test = shuffle_train_test

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        riddles = [self.next_riddle() for i in range(batch_size)]
        riddle_ids = [r.riddle_id for r in riddles]
        max_train_examples = max(len(r.train) for r in riddles)

        inputs = torch.zeros(batch_size, max_train_examples * 2 + 1, 3, 10, 10)  # train input+output = 2 + 1 test input
        targets = torch.zeros(batch_size, 3, 10, 10)
        target_boards = torch.zeros(batch_size, 10, 10, dtype=torch.long)
        color_mapping = self.color_mapping

        # convert board to tensor and map colors
        for i, r in enumerate(riddles):

            all_pairs = list(r.train)
            # all_pairs.extend(r.test)
            all_pairs.append(r.test[0])  # for now only use first test example if multiple are available

            if self.shuffle_train_test:
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
        return inputs, targets, target_boards, riddle_ids


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
    parser.add_argument("--tie_ff", default=False, type=str2bool, help="Tie feed-forward weights of ViT")
    parser.add_argument("--tie_attn", default=False, type=str2bool, help="Tie attention weights of ViT")

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

    parser.add_argument("command", help="train or eval")
    parser.add_argument("--restore", type=str, help="file name of checkpoint to load")
    parser.add_argument("--num_train_examples", default=3, type=int, help="filter riddles based on #train examples")
    parser.add_argument("--max_trainig_riddles", default=-1, type=int, help="maximum number of training files to use")

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
) -> Tuple[float, float, Dict[str, float]]:
    model.eval()

    total_loss = 0
    total_accuracy = 0

    num_total = len(riddle_loader.file_names) if num_batches < 0 else batch_size * num_batches
    num_total = min(len(riddle_loader.file_names), num_total)

    accuracy_by_id = {}

    remaining = num_total
    while remaining > 0:
        N = min(remaining, batch_size)
        remaining -= N

        inputs, targets, target_boards, riddle_ids = riddle_loader.load_batch(N, device)
        b, n, *_ = inputs.shape
        inputs = inputs.view(b, n, -1)
        targets = targets.view(b, -1)

        y = model(inputs)

        loss = loss_fn(y, targets)
        total_loss += loss.item() * N

        t = quantize_output(y, riddle_loader.color_mapping)

        accuracy_per_riddle = (t == target_boards).sum(dim=[1, 2]) / t[0].numel()
        for j, rid in enumerate(riddle_ids):
            accuracy_by_id[rid] = accuracy_per_riddle[j].item()

        accuracy = (t == target_boards).sum() / t.numel()
        total_accuracy += accuracy.item() * N

    return total_loss / num_total, total_accuracy / num_total, accuracy_by_id


@torch.no_grad()
def eval_model_visual(
    device: torch.DeviceObjType, batch_size: int, model, riddle_loader: RiddleLoader
) -> Tuple[float, float]:
    model.eval()

    inputs, targets, target_boards, _ = riddle_loader.load_batch(batch_size, device)

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
        combined.view(-1, 3, 10, 10),
        nrow=num_per_riddle,
        padding=2,
        normalize=False,
        value_range=(0, 1),
    )

    return riddle_grid


def optim_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def run_eval(
    args: argparse.Namespace,
    device: torch.DeviceObjType,
    riddle_loader_eval: RiddleLoader,
    model: nn.Module,
):
    batch_size = args.batch_size
    experiment_name = args.name
    loss_fn = nn.MSELoss()

    if args.save_eval_images:
        num_total = len(riddle_loader_eval.file_names)
        remaining = num_total
        i = 0
        while remaining > 0:
            N = min(remaining, batch_size)
            remaining -= N
            eval_grid = eval_model_visual(device, N, model, riddle_loader_eval)
            torchvision.utils.save_image(eval_grid, f"{experiment_name}_evalrun_{i:03d}.png")
            i += 1

    eval_loss, eval_accuracy, accuracy_by_id = eval_model(device, batch_size, -1, model, riddle_loader_eval, loss_fn)

    for k, v in accuracy_by_id.items():
        print(f"{k}: {v:.2%}")

    print(f"eval loss: {eval_loss:.4e}; eval accuracy: {eval_accuracy:.2%};")


def run_train(
    args: argparse.Namespace,
    device: torch.DeviceObjType,
    riddle_loader: RiddleLoader,
    riddle_loader_eval: RiddleLoader,
    model: nn.Module,
    optimizer,
    lr_scheduler,
    scheduler_cosine,
):
    checkpoint_interval = args.checkpoint_interval
    batch_size = args.batch_size
    max_steps = args.max_steps
    experiment_name = args.name

    loss_fn = nn.MSELoss()

    for step in range(1, max_steps + 1):
        if step % args.eval_interval == 0:
            eval_loss, eval_accuracy, _ = eval_model(
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

            wandb.log(
                {
                    "eval.loss": eval_loss,
                    "eval.accuracy": eval_accuracy,
                    "eval.image": eval_image,
                },
                step=step,
            )

        model.train()
        optimizer.zero_grad()

        batch, targets, target_boards, _ = riddle_loader.load_batch(batch_size, device)
        b, n, *_ = batch.shape
        batch = batch.view(b, n, -1)
        target = targets.view(b, -1)

        y = model(batch)
        loss = loss_fn(y, target)

        loss.backward()
        gn = grad_norm(model.parameters())
        optimizer.step()
        lr_scheduler.step()

        t = quantize_output(y, riddle_loader.color_mapping)
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


def main():
    print(f"Using pytorch version {torch.__version__}")
    args = parse_args()

    print("Effective args:", args)

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
    max_trainig_riddles = args.max_trainig_riddles
    for fn in tqdm(file_names):
        r = load_riddle_from_file(fn)
        if all(
            len(r.train) == args.num_train_examples
            and bp.input.num_cols == 10
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
            if max_trainig_riddles > 0 and len(training_set) > max_trainig_riddles:
                break

    if args.command == "eval":
        eval_file_names = training_set
        train_file_names = None
    else:
        num_train_riddles = len(training_set) * 80 // 100
        eval_file_names = training_set[num_train_riddles:]
        train_file_names = training_set[:num_train_riddles]
        print(f"Num train riddles: {len(train_file_names)}")
        rl = RiddleLoader(file_names=train_file_names, colors_table_name=args.color_table)

    print(f"Num eval riddles: {len(eval_file_names)}")
    riddle_loader_eval = RiddleLoader(
        file_names=eval_file_names,
        colors_table_name=args.color_table,
        shuffle_train_test=False,
    )

    batch, targets, *_ = riddle_loader_eval.load_batch(1, device)
    num_patches = batch.shape[1]
    patch_dim = batch.shape[2] * batch.shape[3] * batch.shape[4]

    chkpt_data = torch.load(args.restore, map_location="cpu") if args.restore is not None else None

    model = ViT_NoEmbed(
        num_patches=num_patches,
        patch_dim=patch_dim,
        num_classes=targets[0].numel(),
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        tie_attn_weights=args.tie_attn,
        tie_ff_weights=args.tie_ff,
    )

    if chkpt_data:
        model.load_state_dict(chkpt_data["model_state_dict"])

    model.to(device)

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

    if chkpt_data:
        optimizer.load_state_dict(chkpt_data["optimizer_state_dict"])
        optim_to(optimizer, device)
        scheduler_cosine.load_state_dict(chkpt_data["cosine_scheduler_state_dict"])
        lr_scheduler.load_state_dict(chkpt_data["warmup_scheduler_state_dict"])

    if args.command == "eval":
        run_eval(args, device, riddle_loader_eval, model)
    elif args.command == "train":
        run_train(
            args,
            device,
            rl,
            riddle_loader_eval,
            model,
            optimizer,
            lr_scheduler,
            scheduler_cosine,
        )
    else:
        print(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
