from typing import List, Set, Tuple
import random
import math
import json
import argparse
from pathlib import Path
import uuid

from arc.interface import Riddle
from arc.utils.dataset import load_riddle_from_file

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import wandb

# from perceiver_pytorch import PerceiverIO
from perceiver_io import PerceiverIO
from warmup_scheduler import GradualWarmupScheduler


class RiddleLoader:
    def __init__(self, file_names: List[Path], function_list: List[str]):
        self.function_list = function_list
        self.function_index = {function_list[i]: i for i in range(len(function_list))}

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

        # get used function names
        graph_fn = fn.parent / "graphs" / (fn.stem + ".graph.json")
        with graph_fn.open("r") as f:
            gd = json.load(f)
            nodes = gd["nodes"]
            function_names = set(n[1] for n in nodes if n[1] != "input")

        return r, function_names

    def load_batch(self, batch_size: int, device: torch.DeviceObjType) -> Tuple:
        riddles = [self.next_riddle() for i in range(batch_size)]

        # find max sizes and training example count
        max_w = max(
            max(t.input.num_cols, t.output.num_cols)
            for r, fx in riddles
            for t in r.train
        )
        max_h = max(
            max(t.input.num_rows, t.output.num_rows)
            for r, fx in riddles
            for t in r.train
        )
        max_num_train = max(len(r.train) for r, fx in riddles for t in r.train)

        inputs = torch.zeros(
            batch_size, max_num_train * 2, max_h, max_w, dtype=torch.long
        )
        input_mask = torch.zeros(
            batch_size, max_num_train * 2, max_h, max_w, dtype=torch.bool
        )
        num_classes = len(self.function_list)
        targets = torch.zeros(batch_size, num_classes, dtype=torch.long)

        # generate values used of position encodings
        train_example_index = inputs.clone()
        io_type = inputs.clone()
        pos_y = inputs.clone()
        pos_x = inputs.clone()

        pos_y[:] = torch.arange(max_h).repeat(max_w, 1).t()
        pos_x[:] = torch.arange(max_w).repeat(max_h, 1)

        targets_list = []
        for i, (r, fn_names) in enumerate(riddles):
            # encode train examples
            for j, t in enumerate(r.train):
                # add input board
                input_board = torch.from_numpy(t.input.np)
                input_shape = input_board.shape

                inputs[i, j * 2, : input_shape[0], : input_shape[1]] = input_board
                input_mask[i, j * 2, : input_shape[0], : input_shape[1]] = True

                # add output board
                output_board = torch.from_numpy(t.output.np)
                output_shape = output_board.shape
                inputs[
                    i, j * 2 + 1, : output_shape[0], : output_shape[1]
                ] = output_board
                input_mask[i, j * 2 + 1, : output_shape[0], : output_shape[1]] = True

                # io_type[i, j * 2] = 0
                io_type[i, j * 2 + 1] = 1
                train_example_index[i, j * 2 + 0] = j
                train_example_index[i, j * 2 + 1] = j

            # function names
            for n in fn_names:
                fn_index = self.function_index[n]
                targets[i, fn_index] = 1
            t = [self.function_index[n] for n in fn_names]
            targets_list.append(t)

        # map tensors to target device
        results = tuple(
            x.to(device)
            for x in (
                inputs,
                input_mask,
                io_type,
                train_example_index,
                pos_x,
                pos_y,
                targets,
            )
        )

        return results + (targets_list,)


class InferFunc(nn.Module):
    def __init__(
        self,
        *,
        num_functions: int = 10,
        dim: int = 256,
        depth: int = 1,
        num_latents: int = 128,
        latent_dim: int = 512,
        max_train_examples: int = 8,
    ):
        super().__init__()

        num_tokens = 10
        max_width, max_height = 32, 32
        num_io_types = 2

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_x_emb = nn.Embedding(max_width, dim)
        self.pos_y_emb = nn.Embedding(max_height, dim)
        self.train_example_index_emb = nn.Embedding(max_train_examples, dim)
        self.io_type_emb = nn.Embedding(num_io_types, dim)

        queries_dim = 64
        self.func_queries = nn.Parameter(torch.randn(num_functions, queries_dim))
        self.to_logits = nn.Linear(num_functions * queries_dim, num_functions)

        self.p = PerceiverIO(
            depth=depth,
            dim=dim,
            queries_dim=queries_dim,
            logits_dim=None,
            num_latents=num_latents,
            latent_dim=latent_dim,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            weight_tie_layers=False,
            decoder_ff=False,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        io_type: torch.Tensor,
        train_example_index: torch.Tensor,
        pos_x: torch.Tensor,
        pos_y: torch.Tensor,
    ) -> torch.Tensor:
        x = self.token_emb(inputs)
        x = x + self.io_type_emb(io_type)
        x = x + self.train_example_index_emb(train_example_index)
        x = x + self.pos_x_emb(pos_x)
        x = x + self.pos_y_emb(pos_y)
        x = x.view(x.size(0), -1, x.size(-1))
        x = self.p(x, mask=mask, queries=self.func_queries)
        x = x.view(x.size(0), -1)
        x = self.to_logits(x)
        return x


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

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_steps", default=5000, type=int)
    parser.add_argument("--warmup", default=500, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--checkpoint_interval", default=25000, type=int)

    parser.add_argument("--dim", default=256, type=int)
    parser.add_argument("--depth", default=1, type=int)
    parser.add_argument("--num_latents", default=128, type=int)
    parser.add_argument("--latent_dim", default=512, type=int)
    parser.add_argument("--max_train_examples", default=8, type=int)
    parser.add_argument("--eval_interval", default=1000, type=int)
    parser.add_argument("--num_eval_batches", default=32, type=int)

    # train_riddle_folder = '/data/synth_riddles/rigid_and_half/rigid_and_half_depth_1_10k/'
    train_riddle_folder = (
        "/mnt/c/code/synth_riddles/rigid_and_half/rigid_and_half_depth_1_10k/"
    )
    parser.add_argument("--riddle_folder", default=train_riddle_folder, type=str)

    parser.add_argument("--wandb", default=False, action="store_true")
    parser.add_argument(
        "--project", default="infer_func", type=str, help="project name for wandb"
    )
    parser.add_argument(
        "--name",
        default="infer_func_" + uuid.uuid4().hex,
        type=str,
        help="wandb experiment name",
    )

    return parser.parse_args()


def load_function_names(riddle_folder: str):
    config_folder_path = Path(riddle_folder) / "config"
    for p in config_folder_path.glob("*.json"):
        with p.open("r") as f:
            cfg = json.load(f)
            if "function_set" in cfg:
                return sorted(cfg["function_set"])
    raise RuntimeError("No configuration with function_set found.")


def count_parameters(module: nn.Module):
    return sum(p.data.nelement() for p in module.parameters())


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
    model: InferFunc,
    riddle_loader: RiddleLoader,
    loss_fn,
) -> float:
    model.eval()

    total_loss = 0
    for i in range(num_batches):
        (
            inputs,
            mask,
            io_type,
            train_example_index,
            pos_x,
            pos_y,
            targets,
            targets_list,
        ) = riddle_loader.load_batch(batch_size, device)
        y = model.forward(inputs, mask, io_type, train_example_index, pos_x, pos_y)
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

    lr = args.lr
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    max_steps = args.max_steps

    fn_list = load_function_names(args.riddle_folder)

    model = InferFunc(
        num_functions=len(fn_list),
        dim=args.dim,
        depth=args.depth,
        num_latents=args.num_latents,
        latent_dim=args.latent_dim,
        max_train_examples=args.max_train_examples,
    )
    print(f"Number of model parameters: {count_parameters(model)}")

    model.to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
        amsgrad=False,
    )

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.max_steps
    )
    lr_scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1.0,
        total_epoch=args.warmup,
        after_scheduler=scheduler_cosine,
    )

    loss_fn = nn.MultiLabelSoftMarginLoss(reduction="mean")
    file_names = list(Path(args.riddle_folder).glob("*.json"))
    random.shuffle(file_names)

    num_train_riddles = len(file_names) * 80 // 100
    eval_file_names = file_names[num_train_riddles:]
    train_file_names = file_names[:num_train_riddles]
    print(f"Num train riddles: {len(train_file_names)}")
    print(f"Num eval riddles: {len(eval_file_names)}")

    riddle_loader_train = RiddleLoader(file_names=train_file_names, function_list=fn_list)
    riddle_loader_eval = RiddleLoader(file_names=eval_file_names, function_list=fn_list)

    checkpoint_interval = args.checkpoint_interval
    for step in range(1, max_steps + 1):

        if step % args.eval_interval == 0:
            eval_loss = eval_model(device, batch_size, args.num_eval_batches, model, riddle_loader_eval, loss_fn)
            print(
                f"step: {step}; eval loss: {eval_loss:.4e};"
            )
            wandb.log({"eval.loss": eval_loss}, step=step)

        model.train()
        optimizer.zero_grad()

        (
            inputs,
            mask,
            io_type,
            train_example_index,
            pos_x,
            pos_y,
            targets,
            targets_list,
        ) = riddle_loader_train.load_batch(batch_size, device)
        y = model.forward(inputs, mask, io_type, train_example_index, pos_x, pos_y)
        loss = loss_fn(y, targets)

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
