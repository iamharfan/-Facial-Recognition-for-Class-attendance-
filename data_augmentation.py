import os
import shutil
import imgaug.augmenters as iaa
from PIL import Image, ImageOps
import numpy as np

def augment_images(input_folder, output_folder, target_count=32):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # List all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Calculate how many augmented images are needed
    num_originals = len(image_files)
    num_augmented = target_count - num_originals
    
    # Define augmentation sequence
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Affine(rotate=(-5, 5)),  # random rotations
        iaa.Multiply((0.7, 1.4)),  # change brightness
        iaa.GaussianBlur(sigma=(0.0, 3.0))  # blur images
    ])
    
    # Copy original images to the output folder
    for image_file in image_files:
        shutil.copy(os.path.join(input_folder, image_file), output_folder)
    
    # Augment images
    for i in range(num_augmented):
        # Select a random image from the original set
        image_file = np.random.choice(image_files)
        image_path = os.path.join(input_folder, image_file)
        
        # Open the image
        image = ImageOps.exif_transpose(Image.open(image_path))
        image_np = np.array(image)
        
        # Apply augmentation
        image_aug = seq(image=image_np)
        
        # Save augmented image
        aug_image = Image.fromarray(image_aug)
        aug_image.save(os.path.join(output_folder, f"xaug_{i}_{image_file}"))

if __name__ == "__main__":
    input_folder = "harfan_76"
    output_folder = "harfan_32"
    augment_images(input_folder, output_folder)