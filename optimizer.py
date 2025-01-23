import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

def create_optimizer_and_scheduler(model, learning_rate=1e-4, warmup_steps=4000, weight_decay=1e-4, d_model=512):
    """
    Create the optimizer and learning rate scheduler for the Transformer model.
    
    Args:
        model (nn.Module): The Transformer model.
        learning_rate (float): Initial learning rate for the optimizer.
        warmup_steps (int): Number of warmup steps for the learning rate scheduler.
        weight_decay (float): Weight decay for the optimizer.
        d_model (int): The dimension of the model (used in the scheduler formula).

    Returns:
        optimizer: Adam optimizer.
        scheduler: LambdaLR learning rate scheduler.
    """
    # Initialize the optimizer
    optimizer = Adam(
        model.parameters(), 
        lr=learning_rate, 
        betas=(0.9, 0.98), 
        eps=1e-9, 
        weight_decay=weight_decay
    )

    # Define the learning rate scheduler
    def lr_schedule(step):
        if step == 0:
            step = 1  # Prevent division by zero

        scale = (d_model ** -0.5)
        return scale * min(step ** -0.5, step * (warmup_steps ** -1.5))


    scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)
    
    return optimizer, scheduler
