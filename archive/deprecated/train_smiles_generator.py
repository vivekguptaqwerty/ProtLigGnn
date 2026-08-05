import argparse
import csv
import json
import random
import struct
import zlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


INPUT_CSV = Path("outputs") / "genai" / "pdbbind_ligands_smiles.csv"
OUTPUT_DIR = Path("outputs") / "genai"
VOCAB_PATH = OUTPUT_DIR / "smiles_vocab.json"
MODEL_PATH = OUTPUT_DIR / "smiles_lstm_generator.pt"
HISTORY_PATH = OUTPUT_DIR / "smiles_generator_history.csv"
LOSS_PLOT_PATH = OUTPUT_DIR / "smiles_generator_loss.png"

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"


class SmilesDataset(Dataset):
    def __init__(self, smiles: Sequence[str], char_to_idx: Dict[str, int], seq_len: int):
        self.sequences = []
        self.seq_len = seq_len
        for item in smiles:
            encoded = [char_to_idx[BOS_TOKEN]]
            encoded.extend(char_to_idx[char] for char in item)
            encoded.append(char_to_idx[EOS_TOKEN])
            self.sequences.append(torch.tensor(encoded, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> torch.Tensor:
        sequence = self.sequences[index]
        window_size = self.seq_len + 1
        if len(sequence) <= window_size:
            return sequence

        start = random.randint(0, len(sequence) - window_size)
        return sequence[start : start + window_size]


class SmilesLSTMGenerator(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, embedding_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        outputs, _ = self.lstm(embedded)
        return self.output(outputs)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a lightweight character-level SMILES LSTM generator.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq_len", type=int, default=96)
    return parser.parse_args()


def load_smiles(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"SMILES CSV not found: {csv_path}")

    smiles = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "smiles" not in reader.fieldnames:
            raise ValueError(f"CSV must contain a 'smiles' column: {csv_path}")

        for row in reader:
            value = row["smiles"].strip()
            if value:
                smiles.append(value)

    if not smiles:
        raise ValueError(f"No SMILES strings found in: {csv_path}")

    return smiles


def build_vocab(smiles: Sequence[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted({char for item in smiles for char in item})
    idx_to_char = {0: PAD_TOKEN, 1: BOS_TOKEN, 2: EOS_TOKEN}
    for offset, char in enumerate(chars, start=3):
        idx_to_char[offset] = char

    char_to_idx = {char: idx for idx, char in idx_to_char.items()}
    return char_to_idx, idx_to_char


def save_vocab(char_to_idx: Dict[str, int], idx_to_char: Dict[int, str], path: Path):
    vocab_payload = {
        "pad_token": PAD_TOKEN,
        "bos_token": BOS_TOKEN,
        "eos_token": EOS_TOKEN,
        "char_to_idx": char_to_idx,
        "idx_to_char": {str(idx): char for idx, char in idx_to_char.items()},
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(vocab_payload, handle, indent=2)


def collate_batch(batch: Sequence[torch.Tensor], pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = [sequence[:-1] for sequence in batch]
    targets = [sequence[1:] for sequence in batch]
    input_batch = pad_sequence(inputs, batch_first=True, padding_value=pad_idx)
    target_batch = pad_sequence(targets, batch_first=True, padding_value=pad_idx)
    return input_batch, target_batch


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        token_count = int((targets != criterion.ignore_index).sum().item())
        total_loss += float(loss.item()) * token_count
        total_tokens += token_count

    return total_loss / max(total_tokens, 1)


def save_history(history: Sequence[Dict[str, float]], path: Path):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss"])
        writer.writeheader()
        writer.writerows(history)


def save_loss_plot(history: Sequence[Dict[str, float]], path: Path):
    width, height = 800, 500
    margin_left, margin_right = 70, 35
    margin_top, margin_bottom = 35, 65
    image = bytearray([255, 255, 255] * width * height)

    def set_pixel(x: int, y: int, color: Tuple[int, int, int]):
        if 0 <= x < width and 0 <= y < height:
            offset = (y * width + x) * 3
            image[offset : offset + 3] = bytes(color)

    def draw_line(x0: int, y0: int, x1: int, y1: int, color: Tuple[int, int, int]):
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            doubled_error = 2 * error
            if doubled_error >= dy:
                error += dy
                x0 += sx
            if doubled_error <= dx:
                error += dx
                y0 += sy

    def draw_circle(cx: int, cy: int, radius: int, color: Tuple[int, int, int]):
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                    set_pixel(x, y, color)

    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom

    axis_color = (45, 45, 45)
    grid_color = (225, 225, 225)
    line_color = (31, 119, 180)

    for i in range(6):
        y = plot_top + round((plot_bottom - plot_top) * i / 5)
        draw_line(plot_left, y, plot_right, y, grid_color)
    draw_line(plot_left, plot_top, plot_left, plot_bottom, axis_color)
    draw_line(plot_left, plot_bottom, plot_right, plot_bottom, axis_color)

    losses = [float(row["loss"]) for row in history]
    min_loss = min(losses)
    max_loss = max(losses)
    if max_loss == min_loss:
        max_loss += 1.0
        min_loss -= 1.0

    points = []
    for index, loss in enumerate(losses):
        x_fraction = index / max(len(losses) - 1, 1)
        y_fraction = (loss - min_loss) / (max_loss - min_loss)
        x = plot_left + round((plot_right - plot_left) * x_fraction)
        y = plot_bottom - round((plot_bottom - plot_top) * y_fraction)
        points.append((x, y))

    for start, end in zip(points, points[1:]):
        draw_line(start[0], start[1], end[0], end[1], line_color)
    for x, y in points:
        draw_circle(x, y, 4, line_color)

    raw_rows = []
    for y in range(height):
        start = y * width * 3
        raw_rows.append(b"\x00" + bytes(image[start : start + width * 3]))
    compressed = zlib.compress(b"".join(raw_rows), level=9)

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk("IHDR".encode("ascii"), struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk("IDAT".encode("ascii"), compressed)
    png += chunk("IEND".encode("ascii"), b"")
    path.write_bytes(png)


def main():
    args = parse_args()
    random.seed(42)
    torch.manual_seed(42)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    smiles = load_smiles(INPUT_CSV)
    char_to_idx, idx_to_char = build_vocab(smiles)
    save_vocab(char_to_idx, idx_to_char, VOCAB_PATH)

    pad_idx = char_to_idx[PAD_TOKEN]
    dataset = SmilesDataset(smiles, char_to_idx, seq_len=args.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, pad_idx),
    )

    device = torch.device("cpu")
    model = SmilesLSTMGenerator(
        vocab_size=len(char_to_idx),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    print(f"loaded SMILES: {len(smiles)}")
    print(f"vocabulary size: {len(char_to_idx)}")
    print(f"device: {device}")
    print(f"epochs: {args.epochs}")
    print(f"sequence window length: {args.seq_len}")

    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, dataloader, optimizer, criterion, device)
        history.append({"epoch": epoch, "loss": loss})
        print(f"epoch {epoch}/{args.epochs} - loss: {loss:.6f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "vocab_size": len(char_to_idx),
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "embedding_dim": 64,
            },
            "training_args": vars(args),
            "vocab_path": str(VOCAB_PATH),
        },
        MODEL_PATH,
    )
    save_history(history, HISTORY_PATH)
    save_loss_plot(history, LOSS_PLOT_PATH)

    print(f"saved vocabulary: {VOCAB_PATH}")
    print(f"saved generator: {MODEL_PATH}")
    print(f"saved history: {HISTORY_PATH}")
    print(f"saved loss plot: {LOSS_PLOT_PATH}")


if __name__ == "__main__":
    main()
