import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from tqdm import tqdm
import os
import nvtx

from model import Transformer, ModelArgs  # assume your model is in model.py


# Dummy dataset for next-token prediction
class DummyDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=128, vocab_size=32000):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = torch.roll(x, shifts=-1, dims=0)  # next-token prediction
        return x, y

def train_model(model, dataloader, optimizer, device, epochs=10):
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for tokens, targets in pbar:
            tokens = tokens.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            start = nvtx.start_range("forward_pass", color="green")
            outputs = model(tokens, start_pos=0)  # [B, T, vocab]
            nvtx.end_range(start)
            
            outputs = outputs.view(-1, outputs.size(-1))  # [B*T, vocab]
            targets = targets.view(-1)  # [B*T]

            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")


def main():
    
    # Model configuration
    # Initialize FairScale model parallel (assumes single-node)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    # initialize_model_parallel(1)  # use 1 if you're not using model parallelism for now
    args = ModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        vocab_size=32000,
        max_batch_size=4,
        max_seq_len=128
    )
    initialize_model_parallel(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(args).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    # DataLoader
    dataset = DummyDataset(num_samples=1000, seq_len=args.max_seq_len, vocab_size=args.vocab_size)
    dataloader = DataLoader(dataset, batch_size=args.max_batch_size, shuffle=True)

    # Train
    train_model(model, dataloader, optimizer, device, epochs=5)

    # Save
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/model.pt")


if __name__ == "__main__":
    main()