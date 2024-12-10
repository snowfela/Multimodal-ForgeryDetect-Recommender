import torch
from .losses import classification_loss, forgery_similarity_loss, diversity_loss

def train_fd_gan(generator_gb, generator_gt, discriminator, dataloader, optimizer, epochs):
    for epoch in range(epochs):
        for source_img, reference_img, label in dataloader:
            # Forward pass through generators
            blended_img, blended_mask = generator_gb(source_img, reference_img)
            transfer_img = generator_gt(source_img, reference_img)

            # Discriminator predictions
            spatial_pred, freq_pred = discriminator(blended_img)

            # Compute losses
            loss_cls = classification_loss(spatial_pred, label)
            loss_sim = forgery_similarity_loss(blended_mask, label)
            loss_div = diversity_loss(generator_gb.mask_filters)

            # Backpropagation
            total_loss = loss_cls + 0.1 * loss_sim + 0.02 * loss_div
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item()}")
