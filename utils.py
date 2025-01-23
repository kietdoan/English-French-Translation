import torch
import os

def create_padding_mask(seq, pad_token=59513, num_heads=8):
    # Expand mask to match multi-head attention dimensions
    return (seq != pad_token).unsqueeze(1).unsqueeze(2).expand(-1, num_heads, -1, -1)

# def create_causal_mask(size, num_heads=8):
#     causal_mask = torch.tril(torch.ones((size, size))).bool()
#     return causal_mask.unsqueeze(0).unsqueeze(0).expand(-1, num_heads, -1, -1)
def create_causal_mask(size, num_heads=8):
    causal_mask = torch.tril(torch.ones((size, size))).bool()
    return causal_mask.unsqueeze(0).unsqueeze(0).expand(-1, num_heads, -1, -1)


def save_checkpoint(model, optimizer, scheduler, epoch, save_path):
    """Save model checkpoint."""
    os.makedirs(save_path, exist_ok=True)
    checkpoint_path = os.path.join(save_path, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]
# def load_checkpoint(path, model, optimizer):
#     checkpoint = torch.load(path)

#     # Load state dict không strict để bỏ qua lỗi kích thước không khớp
#     model.load_state_dict(checkpoint["model_state_dict"], strict=False)

#     # Kiểm tra xem có tham số nào không khớp không
#     missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
#     if missing_keys:
#         print(f"Missing keys: {missing_keys}")
#     if unexpected_keys:
#         print(f"Unexpected keys: {unexpected_keys}")

#     if optimizer:
#         optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

#     return checkpoint["epoch"]
