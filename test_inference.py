import torch
from transformer import build_transformer
from utils import load_checkpoint, create_padding_mask, create_causal_mask

# Define input and label token IDs
test_input = {
    "input_ids": [
        84, 38645, 1869, 12, 5424, 4, 317, 21, 22273, 7, 22284, 0
    ],
    "labels": [
        469, 1603, 35900, 51, 27, 279, 139, 8, 906, 9915, 20, 6, 27953, 6429, 0
    ]
}

# Model parameters (ensure these match the trained model's parameters)
src_vocab_size = 59514
tgt_vocab_size = 59514
max_length = 512
d_model = 512
num_layers = 6
num_heads = 8
d_ff = 2048
dropout = 0.1
pad_token = 59513
end_token = 0

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading trained model...")
checkpoint_path = "checkpoints/checkpoint_epoch_10.pt"
model = build_transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    src_seq_len=max_length,
    tgt_seq_len=max_length,
    d_model=d_model,
    N=num_layers,
    head=num_heads,
    d_ff=d_ff,
    dropout=dropout
).to(device)
# print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
print("Running one batch of random data to initialize model parameters...")
src_dummy = torch.randint(0, src_vocab_size, (1, max_length)).to(device)
tgt_dummy = torch.randint(0, tgt_vocab_size, (1, max_length)).to(device)
src_mask_dummy = torch.ones(1, 8, 1, max_length).to(device)
tgt_mask_dummy = torch.ones(1, 8, max_length, max_length).to(device)

with torch.no_grad():
    model(src_dummy, tgt_dummy, src_mask_dummy, tgt_mask_dummy)
start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
print(f"Model loaded from checkpoint. Resuming from epoch {start_epoch}")

model.eval()

# Prepare input tensor
input_ids = test_input["input_ids"] + [pad_token] * (max_length - len(test_input["input_ids"]))
src_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

# Create masks
src_mask = create_padding_mask(src_tensor, pad_token=pad_token)

# Perform inference
with torch.no_grad():
    encoder_output = model.encode(src_tensor, src_mask)

    tgt_tokens = [pad_token]
    for _ in range(max_length):
        tgt_tensor = torch.tensor([tgt_tokens], dtype=torch.long).to(device)
        tgt_pad_mask = create_padding_mask(tgt_tensor, pad_token=pad_token)
        tgt_causal_mask = create_causal_mask(tgt_tensor.size(1)).to(device)
        tgt_mask = tgt_pad_mask & tgt_causal_mask

        # Decode
        decoder_output = model.decode(encoder_output, src_mask, tgt_tensor, tgt_mask)
        logits = model.project(decoder_output)

        # Get predicted token
        next_token = logits[:, -1, :].argmax(dim=-1).item()
        tgt_tokens.append(next_token)

        # Stop if the <END> token is generated
        if next_token == end_token:
            break

print(f"Input Token IDs: {test_input['input_ids']}")
print(f"Translated Token IDs: {tgt_tokens}")
