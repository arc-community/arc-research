import math
from typing import List, Set, Tuple
import random
import argparse
from pathlib import Path
import uuid
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from arc.interface import Riddle
from arc.utils.dataset import load_riddle_from_file

import wandb

# from perceiver_pytorch import PerceiverIO
from perceiver_io import Attention, FeedForward, PerceiverIO, PreNorm
from warmup_scheduler import GradualWarmupScheduler


class RiddleLoader:
    def __init__(self, file_names: List[Path], max_board_size: Tuple[int, int], color_permutation: bool = False):
        self.file_names = file_names
        if len(self.file_names) == 0:
            raise RuntimeError("No riddles found.")
        self.order = list(range(len(self.file_names)))
        self.shuffle()
        self.max_board_size = max_board_size
        self.color_permutation = color_permutation

    def shuffle(self) -> None:
        random.shuffle(self.order)
        self.next_index = 0

    def next_riddle(self) -> Tuple[Riddle, Set[str]]:
        if self.next_index >= len(self.file_names):
            self.shuffle()
        fn = self.file_names[self.order[self.next_index]]
        self.next_index += 1
        return load_riddle_from_file(fn)

    def load_batch(self, batch_size: int, device: torch.DeviceObjType) -> Tuple:
        riddles = [self.next_riddle() for i in range(batch_size)]

        w, h = self.max_board_size
        inputs = torch.zeros(batch_size, 1, h, w, dtype=torch.long)
        input_mask = torch.zeros(batch_size, 1, h, w, dtype=torch.bool)
        board_sizes = torch.zeros(batch_size, 2, dtype=torch.long)

        # generate values used of position encodings
        pos_y = inputs.clone()
        pos_x = inputs.clone()

        pos_y[:] = torch.arange(h).repeat(w, 1).t()
        pos_x[:] = torch.arange(w).repeat(h, 1)

        for i, r in enumerate(riddles):

            # optional random color permutation augmentation
            if self.color_permutation:
                c = list(range(1, 10))
                random.shuffle(c)
                c = [0] + c

            all_pairs = list(r.train)
            all_pairs.extend(r.test)
            p = random.choice(all_pairs)
            board = random.choice([p.input, p.output])

            if not self.color_permutation:
                board = torch.from_numpy(board.np)
            else:
                board = torch.tensor([c[x] for x in board.data_flat]).view(board.num_rows, board.num_cols)
            board_shape = board.shape
            board_sizes[i, 0] = board_shape[0]  # rows
            board_sizes[i, 1] = board_shape[1]  # cols

            inputs[i, 0, : board_shape[0], : board_shape[1]] = board
            input_mask[i, 0, : board_shape[0], : board_shape[1]] = True

        # map tensors to target device
        results = tuple(x.to(device) for x in (inputs, input_mask, pos_x, pos_y, board_sizes))

        return results


class BoardDesc(nn.Module):
    def __init__(
        self,
        *,
        dim: int = 256,
        depth: int = 1,
        num_latents: int = 128,
        latent_dim: int = 512,
        max_num_desc_tokens: int = 32,
        desc_queries_dim: int = 64,
        decoder_depth: int = 2,
        separate_out_pos_embeddings: bool = True,
        max_width: int = 30,
        max_height: int = 30,
        num_colors: int = 10,
        initializer_range: float = 0.02
    ):
        super().__init__()

        self.max_width = max_height
        self.max_height = max_height
        self.num_colors = num_colors
        self.initializer_range = initializer_range

        self.token_emb = nn.Embedding(num_colors, dim)
        self.pos_x_emb = nn.Embedding(max_width, dim)
        self.pos_y_emb = nn.Embedding(max_height, dim)

        self.max_num_desc_tokens = max_num_desc_tokens
        self.desc_queries = nn.Parameter(torch.zeros(max_num_desc_tokens, desc_queries_dim))

        if separate_out_pos_embeddings:
            self.out_pos_x_emb = nn.Embedding(max_width, dim)
            self.out_pos_y_emb = nn.Embedding(max_height, dim)
        else:
            self.out_pos_x_emb = self.pos_x_emb
            self.out_pos_y_emb = self.pos_y_emb

        self.p = PerceiverIO(
            depth=depth,
            dim=dim,
            queries_dim=desc_queries_dim,
            logits_dim=None,
            num_latents=num_latents,
            latent_dim=latent_dim,
            cross_heads=4,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            weight_tie_layers=False,
            decoder_ff=False,
            initializer_range=1.0
        )

        self.decoder_cross_attn = PreNorm(
            dim, Attention(dim, desc_queries_dim, heads=1, dim_head=64), context_dim=desc_queries_dim
        )

        self.decoder_layers = nn.ModuleList([])
        for i in range(decoder_depth):
            self.decoder_layers.append(
                nn.ModuleList([PreNorm(dim, Attention(dim, heads=4, dim_head=64)), PreNorm(dim, FeedForward(dim))])
            )
        self.to_logits = nn.Linear(dim, num_colors)

        # board size decoder
        self.out_board_size_qry = nn.Parameter(torch.zeros(1, dim))
        self.to_height_logits = nn.Linear(dim, max_height)
        self.to_width_logits = nn.Linear(dim, max_width)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.out_board_size_qry, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.desc_queries, mean=0.0, std=self.initializer_range)
        
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=self.initializer_range)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        pos_x: torch.Tensor,
        pos_y: torch.Tensor,
        num_use_desc_tokens: List[int],
    ) -> List[torch.Tensor]:

        B = inputs.shape[0]
        device = inputs.device

        x = self.token_emb(inputs)

        x = x + self.pos_x_emb(pos_x)
        x = x + self.pos_y_emb(pos_y)
        x = x.view(B, -1, x.size(-1))

        bottleneck_embedding = self.p(x, mask=mask, queries=self.desc_queries)

        ys = []
        zs = []
        for nd in num_use_desc_tokens:
            assert nd > 0 and nd <= self.max_num_desc_tokens

            ctx = bottleneck_embedding[:, :nd]

            # create 30x30 positional encoding output queries
            x_pos = torch.arange(self.max_width, device=device).repeat(self.max_height).view(-1)
            y_pos = torch.arange(self.max_height, device=device).unsqueeze(-1).repeat(1, self.max_width).view(-1)

            out_pos_queries = self.out_pos_x_emb(x_pos) + self.out_pos_y_emb(y_pos)

            decoder_queries = torch.concat((out_pos_queries, self.out_board_size_qry))
            decoder_queries = decoder_queries.repeat(B, 1, 1).to(device)

            # cross attend bottleneck embeddings
            x = self.decoder_cross_attn(decoder_queries, context=ctx)

            # decoder
            for self_attn, self_ff in self.decoder_layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

            # project to board size logits
            size_decoder_output = x[:, -1]
            h_logits = self.to_height_logits(size_decoder_output)
            w_logits = self.to_width_logits(size_decoder_output)
            zs.append((h_logits, w_logits))

            # project to board colors
            x = x[:, :-1]
            x = self.to_logits(x)
            ys.append(x)

        return ys, zs


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

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_steps", default=100000, type=int)
    parser.add_argument("--warmup", default=500, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--checkpoint_interval", default=-1, type=int)

    parser.add_argument("--dim", default=256, type=int)
    parser.add_argument("--depth", default=8, type=int)
    parser.add_argument("--num_latents", default=128, type=int)
    parser.add_argument("--latent_dim", default=512, type=int)
    parser.add_argument("--desc_queries_dim", default=64, type=int)
    parser.add_argument("--decoder_depth", default=2, type=int)

    parser.add_argument("--eval_interval", default=500, type=int)
    parser.add_argument("--num_eval_batches", default=32, type=int)

    train_riddle_folder = "/data/synth_riddles/rigid_plus2/rigid_varsize_depth_1_100k/"
    parser.add_argument("--riddle_folder", default=train_riddle_folder, type=str)
    parser.add_argument("--color_permutation", default=True, type=str2bool, help="Flag color permutation")

    parser.add_argument("--wandb", default=False, action="store_true")
    parser.add_argument("--project", default="board_desc", type=str, help="project name for wandb")
    parser.add_argument(
        "--name",
        default="board_desc_" + uuid.uuid4().hex,
        type=str,
        help="wandb experiment name",
    )

    return parser.parse_args()


def count_parameters(module: nn.Module):
    return sum(p.data.nelement() for p in module.parameters())


@torch.no_grad()
def grad_norm(model_params):
    sqsum = 0.0
    for p in model_params:
        if p.grad is not None:
            sqsum += (p.grad**2).sum().item()
    return math.sqrt(sqsum)


@torch.no_grad()
def accuracy(y_logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
    p = torch.argmax(y_logits, dim=-1)
    mask = mask.view_as(p)
    targets = targets.view_as(p).masked_fill(mask.bitwise_not(), -1)
    matches = p.eq(targets).sum(dim=-1)
    counts = mask.sum(dim=-1)
    per_example_accuracy = matches / counts
    accuracy = per_example_accuracy.mean()
    return accuracy


@torch.no_grad()
def eval_model(
    device: torch.DeviceObjType,
    batch_size: int,
    num_batches: int,
    model: BoardDesc,
    riddle_loader: RiddleLoader,
    loss_fn,
    num_use_desc_tokens: List[int],
) -> float:
    model.eval()

    total_loss = torch.zeros(len(num_use_desc_tokens), device=device)
    total_size_loss = torch.zeros_like(total_loss)
    total_acc = torch.zeros_like(total_loss)

    for i in range(num_batches):
        (inputs, mask, pos_x, pos_y, board_sizes) = riddle_loader.load_batch(batch_size, device)

        # use a variable number of elements from the bottleneck representation as input for reconstruction
        ys, zs = model.forward(inputs, mask, pos_x, pos_y, num_use_desc_tokens=num_use_desc_tokens)

        targets = inputs.clone()
        targets[mask == False] = -1  # mask out unused values

        for j, (y, z) in enumerate(zip(ys, zs)):
            gt_h, gt_w = board_sizes[:, 0], board_sizes[:, 1]
            lh = loss_fn(z[0].view(-1, model.max_height), gt_h.view(-1) - 1)
            lw = loss_fn(z[1].view(-1, model.max_width), gt_w.view(-1) - 1)
            total_size_loss[j] += lh + lw
            l = loss_fn(y.view(-1, 10), targets.view(-1))
            total_loss[j] += l
            a = accuracy(y, targets, mask)
            total_acc[j] += a

    return total_loss / num_batches, total_acc / num_batches, total_size_loss / num_batches


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

    lr = args.lr
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    max_steps = args.max_steps

    file_names = list(Path(args.riddle_folder).glob("*.json"))
    random.shuffle(file_names)

    num_train_riddles = len(file_names) * 80 // 100
    eval_file_names = file_names[num_train_riddles:]
    train_file_names = file_names[:num_train_riddles]
    print(f"Num train riddles: {len(train_file_names)}")
    print(f"Num eval riddles: {len(eval_file_names)}")

    riddle_loader_train = RiddleLoader(
        file_names=train_file_names, max_board_size=(30, 30), color_permutation=args.color_permutation
    )
    riddle_loader_eval = RiddleLoader(file_names=eval_file_names, max_board_size=(30, 30))

    model = BoardDesc(
        dim=args.dim,
        depth=args.depth,
        num_latents=args.num_latents,
        latent_dim=args.latent_dim,
        desc_queries_dim=args.desc_queries_dim,
        decoder_depth=args.decoder_depth,
        max_num_desc_tokens=64,
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

    loss_fn = nn.CrossEntropyLoss(reduction="mean", ignore_index=-1)

    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_steps)
    lr_scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1.0,
        total_epoch=args.warmup,
        after_scheduler=scheduler_cosine,
    )

    checkpoint_interval = args.checkpoint_interval
    num_desc_tokens = [4, 8, 16, 32, 64]
    for step in range(1, max_steps + 1):

        if step % args.eval_interval == 0:
            eval_loss, eval_acc, size_loss = eval_model(
                device,
                batch_size,
                args.num_eval_batches,
                model,
                riddle_loader_eval,
                loss_fn,
                num_use_desc_tokens=num_desc_tokens,
            )
            print(
                f"step: {step}; eval loss: {eval_loss.mean():.4e}; eval accuracy: {eval_acc.mean():.2%}; size loss: {size_loss.mean():.4e};"
            )

            eval_log_data = {
                "eval.loss.mean": eval_loss.mean(),
                "eval.accuracy.mean": eval_acc.mean(),
                "eval.size_loss.mean": size_loss.mean(),
            }
            for i, cnt in enumerate(num_desc_tokens):
                eval_log_data[f"eval.loss.{cnt}"] = eval_loss[i]
                eval_log_data[f"eval.accuracy.{cnt}"] = eval_acc[i]
                eval_log_data[f"eval.size_loss.{cnt}"] = size_loss[i]
            wandb.log(eval_log_data, step=step)

        model.train()
        optimizer.zero_grad()

        (inputs, mask, pos_x, pos_y, board_sizes) = riddle_loader_train.load_batch(batch_size, device)

        # use a variable number of elements from the bottleneck representation as input for reconstruction
        ys, zs = model.forward(inputs, mask, pos_x, pos_y, num_use_desc_tokens=num_desc_tokens)

        targets = inputs.clone()
        targets[mask == False] = -1  # mask out unused values

        train_loss = torch.zeros(len(num_desc_tokens), device=device)
        train_acc = torch.zeros_like(train_loss)
        size_loss = torch.zeros_like(train_loss)

        for i, (y, z) in enumerate(zip(ys, zs)):
            gt_h, gt_w = board_sizes[:, 0], board_sizes[:, 1]
            lh = loss_fn(z[0].view(-1, model.max_height), gt_h.view(-1) - 1)
            lw = loss_fn(z[1].view(-1, model.max_width), gt_w.view(-1) - 1)
            size_loss[i] = lh + lw
            train_loss[i] = loss_fn(y.view(-1, 10), targets.view(-1))
            train_acc[i] = accuracy(y, targets, mask)

        loss = train_loss.sum() + 0.1 * size_loss.sum()
        loss.backward()
        gn = grad_norm(model.parameters())
        optimizer.step()
        lr_scheduler.step()

        if step % 10 == 0:
            print(
                f"step: {step}; train loss: {train_loss.sum().item():.4e}; size loss: {size_loss.sum().item():.4e}; lr: {lr_scheduler.get_last_lr()[0]:.3e}; grad_norm: {gn:.3e};"
            )
            for i, cnt in enumerate(num_desc_tokens):
                print(
                    f"loss.{cnt:02d}: {train_loss[i].item():.4e}; acc: {train_acc[i].item():.2%}; size loss: {size_loss[i].item():.4e};"
                )

        train_log_data = {
            "train.loss.mean": train_loss.mean(),
            "train.size_loss.mean:": size_loss.mean(),
            "train.accuracy.mean": train_acc.mean(),
            "train.lr": lr_scheduler.get_last_lr()[0],
            "train.grad_norm": gn,
        }
        for i, cnt in enumerate(num_desc_tokens):
            train_log_data[f"train.loss.{cnt}"] = train_loss[i]
            train_log_data[f"train.accuracy.{cnt}"] = train_acc[i]
            train_log_data[f"train.size_loss.{cnt}"] = size_loss[i]
        wandb.log(train_log_data, step=step)

        if checkpoint_interval > 0 and step % checkpoint_interval == 0:
            # write model_checkpoint
            experiment_name = args.name
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
