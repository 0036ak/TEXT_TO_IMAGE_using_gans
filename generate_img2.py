
import json
import os

import numpy as np
import torch
from PIL import Image

from models.generator import Generator
from utils.data_loader import DataLoader

# Optional: import json if you want to save metadata
# import json


# --- Configuration (match your train.py settings) ---
latent_dim = 100
embedding_dim = 256
caption_encoder_hidden_size = 256
image_shape = (3, 64, 64) # (channels, height, width)
model_path = './saved_models/generator_final.pth' # Path to your trained model
output_dir = './generated_images' # Directory to save generated images

# Data Loader paths (must match what was used for training vocabulary)
image_dir_for_vocab = './data/Images'
caption_file_for_vocab = './data/captions/captions.txt'

max_caption_length = 50 # This should match the max_len used during training

# Example captions to generate images for
example_captions = [
    "glass with water",
    "a cat sitting on a chair",
    "a dog playing with a ball",
    "a beautiful sunset over the mountains",
    "a person riding a bicycle in the park",
    "a plate of delicious food",
    "a car driving on a road",
]

# --- Main Generation Logic ---
def generate():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for generation: {device}")

    # Initialize DataLoader (only to get vocabulary for encoding captions)
    try:
        data_loader = DataLoader(
            image_dir=image_dir_for_vocab,
            caption_file=caption_file_for_vocab,
            image_size=image_shape[1:],
            min_word_freq=5
        )
        vocab_size = data_loader.vocab_size
        print(f"DataLoader initialized for vocabulary. Vocab size: {vocab_size}")
    except Exception as e:
        print(f"Error initializing DataLoader for vocabulary: {e}")
        print("Please ensure image_dir_for_vocab and caption_file_for_vocab are correct and point to your dataset.")
        return

    # Initialize Generator model
    generator = Generator(
        noise_dim=latent_dim,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        caption_encoder_hidden_size=caption_encoder_hidden_size,
        output_channels=image_shape[0]
    ).to(device)

    # Load trained model weights
    if os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path, map_location=device))
        generator.eval() # Set to evaluation mode
        print(f"Successfully loaded generator from {model_path}")
    else:
        print(f"Error: Generator model not found at {model_path}. Please check the path.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving generated images to: {output_dir}")

    # Optional: Prepare a list to store generated image paths and their original captions
    generated_data = []

    with torch.no_grad(): # Disable gradient calculations for inference
        for i, caption_text in enumerate(example_captions):
            # Generate random noise vector
            z = torch.randn(1, latent_dim, device=device) # Batch size of 1 for single image generation

            # Encode caption
            encoded_caption_np = data_loader.tokenize_and_encode_caption(caption_text, max_len=max_caption_length)
            encoded_caption_tensor = torch.tensor(encoded_caption_np, dtype=torch.long).unsqueeze(0).to(device) # Add batch dimension

            # Generate image
            generated_img = generator(z, encoded_caption_tensor)

            # Post-process image: denormalize from [-1, 1] to [0, 255]
            generated_img = (generated_img.squeeze(0).cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
            generated_img = np.transpose(generated_img, (1, 2, 0)) # Change from (C, H, W) to (H, W, C) for PIL

            # Convert to PIL Image and save
            img_filename = f'generated_image_{i+1}.png'
            img_filepath = os.path.join(output_dir, img_filename)
            img_pil = Image.fromarray(generated_img)
            img_pil.save(img_filepath)
            print(f"Generated image for '{caption_text}' saved as {img_filename}")

            # Store the generated image path and its original caption
            generated_data.append({
                'image_path': img_filepath,
                'original_caption': caption_text
            })

    print("Image generation complete!")

    # Optional: Save metadata to a JSON file for easy access in the next stage
    if generated_data:
        metadata_filepath = os.path.join(output_dir, 'generated_images_metadata.json')
        with open(metadata_filepath, 'w') as f:
            json.dump(generated_data, f, indent=4)
        print(f"Generated images metadata saved to {metadata_filepath}")

    return generated_data # Return data for programmatic use if desired

if __name__ == '__main__':
    # You can call generate() and get the data back for the next stage
    generated_results = generate()
    # Now 'generated_results' contains list of dicts:
    # [{'image_path': '...', 'original_caption': '...'}, ...]
    # You would pass this to your high-fidelity model script