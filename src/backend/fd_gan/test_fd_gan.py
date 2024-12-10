import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("C:/Users/sivan/Downloads/FinalYearProject/e-commerce-store/src/backend")
from fd_gan.generators import BlendGenerator, TransferGenerator
from fd_gan.discriminator import Discriminator
#from fd_gan.utils import load_real_data  # Uncomment if using real images

# Dummy image tensor for testing
dummy_image_tensor = torch.rand((1, 3, 256, 256))
print("Dummy image tensor created:", dummy_image_tensor.shape)

def generate_dummy_data(image_size=(3, 256, 256)):
    source_image = torch.rand((1, *image_size))  # Random tensor for source image
    reference_image = torch.rand((1, *image_size))  # Random tensor for reference image
    return source_image, reference_image

def show_image(tensor, title="Image"):
    """
    Utility function to display a tensor as an image.
    Converts the tensor from a range of [0, 1] to [0, 255] and shows it using matplotlib.
    """
    tensor = tensor.squeeze(0)  # Remove the batch dimension
    tensor = tensor.permute(1, 2, 0).detach().cpu().numpy()  # Convert to HWC format for visualization
    tensor = np.clip(tensor, 0, 1)  # Ensure the values are between 0 and 1
    plt.imshow(tensor)
    plt.title(title)
    plt.axis('off')  # Hide axis
    plt.show()

def test_blend_generator():
    print("Testing Blend Generator...")
    generator = BlendGenerator(num_masks=4, feature_channels=256)
    feature_map = torch.rand((1, 256, 64, 64))  # Simulate feature map
    source_image, reference_image = generate_dummy_data()
    blended_image, blended_mask = generator(feature_map, source_image, reference_image)    
    # Debugging print statements for shape inspection
    print(f"blended_mask shape: {blended_mask.shape}")
    print(f"source_image shape: {source_image.shape}")    
    # Show the blended image and mask
    show_image(blended_image[0], title="Blended Image")
    show_image(blended_mask[0], title="Blended Mask")
    # Verify shapes match
    assert blended_image.shape == source_image.shape, "Blended image size mismatch"
    assert blended_mask.shape[1:] == source_image.shape[1:], f"Mask size mismatch: {blended_mask.shape[1:]} vs {source_image.shape[1:]}"
    print("Blend Generator passed!")

def test_transfer_generator():
    print("Testing Transfer Generator...")
    generator = TransferGenerator(style_channels=256, content_channels=256)
    source_image, reference_image = generate_dummy_data()
    synthesized_image = generator(source_image, reference_image)   
    # Show the synthesized image
    show_image(synthesized_image[0], title="Synthesized Image")
    # Verify synthesized image size matches the source image
    assert synthesized_image.shape == source_image.shape, "Synthesized image size mismatch"
    print("Transfer Generator passed!")

def test_discriminator():
    print("Testing Discriminator...")
    discriminator = Discriminator()
    image = torch.rand((1, 3, 256, 256))  # Random image input
    spatial_pred, frequency_pred = discriminator(image)      
    # Ensure the discriminator outputs have the correct shape
    assert spatial_pred.shape[0] == 1, "Discriminator spatial output size mismatch"
    print("Discriminator passed!")

if __name__ == "__main__":
    test_blend_generator()
    test_transfer_generator()
    test_discriminator()
