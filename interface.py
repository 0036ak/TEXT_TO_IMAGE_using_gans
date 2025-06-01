# e:\hello\inference_app.py

import os  # Make sure os is imported for directory creation

import numpy as np
import torch
from PIL import Image

from models.generator import Generator
# Import your DataLoader and Generator classes
from utils.data_loader import DataLoader

# --- Configuration (These MUST match your training setup in train.py) ---
latent_dim = 100
embedding_dim = 256
caption_encoder_hidden_size = 256
image_size = (64, 64) # The output size of your Generator
max_caption_length = 50 # Max sequence length for tokenization

# Path to your saved generator model
SAVED_GENERATOR_PATH = 'saved_models/generator_final.pth'

# Paths to your data (needed by DataLoader for vocabulary)
# The image_dir doesn't need to actually contain images for inference, but DataLoader requires a path.
IMAGE_DIR_FOR_VOCAB = 'data/images'
CAPTION_FILE_FOR_VOCAB = 'data/captions/captions.txt'

# Output directory for generated images
OUTPUT_IMAGE_DIR = 'generated_output'
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True) # Create the output directory if it doesn't exist

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for inference: {device}")

# --- Initialize DataLoader (to get the vocabulary) ---
# We instantiate DataLoader solely to get access to its word_to_idx and tokenize_and_encode_caption methods,
# which are essential for converting user text into numerical IDs.
print("Initializing DataLoader to build vocabulary...")
data_loader = DataLoader(
    image_dir=IMAGE_DIR_FOR_VOCAB,
    caption_file=CAPTION_FILE_FOR_VOCAB,
    image_size=image_size, # Can be dummy, but must match expected if you were loading images
    min_word_freq=5 # IMPORTANT: This must match the min_word_freq used during training
)
print(f"Vocabulary loaded with size: {data_loader.vocab_size}")

# --- Initialize and Load Generator Model ---
print("Initializing Generator model...")
generator = Generator(
    noise_dim=latent_dim,
    vocab_size=data_loader.vocab_size, # Get the actual vocabulary size
    embedding_dim=embedding_dim,
    caption_encoder_hidden_size=caption_encoder_hidden_size,
    output_channels=3 # Assuming RGB images
).to(device)

if os.path.exists(SAVED_GENERATOR_PATH):
    print(f"Loading trained generator weights from: {SAVED_GENERATOR_PATH}")
    # Load the state_dict
    generator.load_state_dict(torch.load(SAVED_GENERATOR_PATH, map_location=device))
    # Set model to evaluation mode (important for inference)
    generator.eval()
    print("Generator weights loaded successfully. Model ready for inference.")
else:
    print(f"Error: Trained Generator model not found at {SAVED_GENERATOR_PATH}.")
    print("Please ensure you have trained your model using train.py and saved it.")
    exit() # Exit if the model isn't found

# --- Image Generation Function ---
def generate_image(text_description, num_images_to_generate=1):
    """
    Generates an image (or images) from a text description using the loaded GAN.
    """
    print(f"\nGenerating {num_images_to_generate} image(s) for: '{text_description}'...")

    # 1. Prepare the text input
    # Tokenize and encode the input text using the DataLoader's vocabulary
    encoded_caption_np = data_loader.tokenize_and_encode_caption(text_description, max_len=max_caption_length)
    
    # Repeat the encoded caption for the desired number of images in the batch
    batch_encoded_captions_tensor = torch.LongTensor(np.tile(encoded_caption_np, (num_images_to_generate, 1))).to(device)

    # 2. Generate random noise vector
    z = torch.randn(num_images_to_generate, latent_dim).to(device)

    # 3. Generate the image(s) using the Generator
    with torch.no_grad(): # Disable gradient calculations for inference, saves memory and speeds up
        generated_images_tensor = generator(z, batch_encoded_captions_tensor)

    # 4. Post-process the generated images
    # Scale from [-1, 1] (Tanh output) to [0, 255] for image saving
    generated_images_tensor = (generated_images_tensor + 1) / 2.0 # Scale to [0, 1]
    
    # Permute dimensions: (batch_size, C, H, W) -> (batch_size, H, W, C) for PIL
    generated_images_np = generated_images_tensor.permute(0, 2, 3, 1).cpu().numpy()

    output_pil_images = []
    for i in range(num_images_to_generate):
        # Convert to 8-bit integer and create PIL Image
        img_array = (generated_images_np[i] * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        output_pil_images.append(img)
    
    print("Image generation complete.")
    return output_pil_images

# --- User Interface Loop ---
if __name__ == "__main__":
    print("\n--- Text-to-Image Generation Interface ---")
    print("Type a description (e.g., 'a red car on a street')")
    print("Type 'q' or 'quit' to exit.")

    while True:
        text_input = input("\nEnter text description: ").strip()
        if text_input.lower() in ['q', 'quit']:
            break

        try:
            num_str = input("How many images to generate? (Default: 1): ").strip()
            num_images = int(num_str) if num_str.isdigit() and int(num_str) > 0 else 1
            
            generated_imgs = generate_image(text_input, num_images)
            
            # Save the generated images
            for i, img in enumerate(generated_imgs):
                # Sanitize filename by replacing spaces with underscores and removing special chars
                sanitized_text = "".join(c for c in text_input if c.isalnum() or c == ' ').replace(' ', '_')
                file_name = f"{sanitized_text}_{i+1}.png"
                save_path = os.path.join(OUTPUT_IMAGE_DIR, file_name)
                img.save(save_path)
                print(f"Saved image to: {save_path}")
            
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure your model is trained and paths are correct.")

    print("\nExiting. Happy generating!")