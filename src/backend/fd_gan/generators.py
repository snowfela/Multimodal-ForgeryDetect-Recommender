import torch
import torch.nn as nn
import torch.nn.functional as F  # For interpolation

class BlendGenerator(nn.Module):
    def __init__(self, num_masks=8, feature_channels=256):
        super(BlendGenerator, self).__init__()
        self.num_masks = num_masks
        # Define mask generators (1 output channel per mask)
        self.mask_filters = nn.ModuleList([
            nn.Conv2d(feature_channels, 1, kernel_size=1) for _ in range(num_masks)
        ])
    
    def forward(self, feature_map, source_img, reference_img):
        # Generate masks for manipulated regions (each mask is 1 channel)
        masks = [torch.sigmoid(f(feature_map)) for f in self.mask_filters]
        
        # Average the masks to get a single blended mask
        blended_mask = torch.clip(sum(masks) / self.num_masks, 0, 1)

        # Debugging: print out the mask's shape before resizing
        print(f"Original blended_mask shape: {blended_mask.shape}")
        
        # Ensure the blended_mask has the same spatial dimensions as source_img and reference_img
        if blended_mask.shape[2:] != source_img.shape[2:]:
            blended_mask = F.interpolate(
                blended_mask, size=source_img.shape[2:], mode='bilinear', align_corners=False
            )
        
        # Expand the mask to match the number of channels in the source and reference images
        # The expanded mask should have 3 channels
        blended_mask = blended_mask.repeat(1, 3, 1, 1)  # Repeat across the channel dimension
        
        # Debugging: print out the mask's shape after resizing and expanding
        print(f"Expanded blended_mask shape: {blended_mask.shape}")
        
        # Blend source and reference images using the blended mask
        blended_image = blended_mask * reference_img + (1 - blended_mask) * source_img
        
        return blended_image, blended_mask

class TransferGenerator(nn.Module):
    def __init__(self, style_channels=256, content_channels=256):
        super(TransferGenerator, self).__init__()
        # Style and content encoders
        self.style_encoder = nn.Sequential(
            nn.Conv2d(3, style_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.content_encoder = nn.Sequential(
            nn.Conv2d(3, content_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        # Decoder to combine style and content features
        self.decoder = nn.Sequential(
            nn.Conv2d(content_channels, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # To get values between -1 and 1
        )
    def forward(self, source_img, reference_img):
        # Extract features from source and reference images
        style_features = self.style_encoder(reference_img)
        content_features = self.content_encoder(source_img)
        # Combine style and content features
        combined_features = content_features + style_features
        # Decode the combined features to generate the synthesized image
        synthesized_image = self.decoder(combined_features)
        return synthesized_image
