# e:\hello\utils\data_loader.py

import json
import os
import random  # Added for random caption selection in get_batch
from collections import Counter

import numpy as np
import pandas as pd  # Added for CSV parsing
from PIL import Image


class DataLoader:
    def __init__(self, image_dir, caption_file, image_size=(64, 64), min_word_freq=5):
        print("DataLoader: Initializing...")
        self.image_dir = image_dir
        self.image_size = image_size
        self.min_word_freq = min_word_freq # Store min_word_freq for build_vocabulary

        print("DataLoader: Loading and mapping captions data (from CSV)...")
        # self.captions will now be a dictionary: {image_filename: [caption1, caption2, ...]}
        self.captions = self._load_and_map_flickr_captions(caption_file)
        
        # Get a list of all unique image filenames that have captions
        self.image_filenames = sorted(list(self.captions.keys()))
        
        # This will be used in get_batch to iterate over available images
        # The 'ids' concept from COCO is no longer directly relevant with Flickr8k format
        # self.ids = list(range(len(self.image_filenames))) # Not strictly needed as we use image_filenames directly

        print("DataLoader: Building vocabulary...")
        self.vocab_obj = self.build_vocabulary() 
        self.word_to_idx = self.vocab_obj['word_to_idx']
        self.idx_to_word = self.vocab_obj['idx_to_word']
        self.vocab_size = len(self.word_to_idx)
        print(f"Vocabulary size: {self.vocab_size}")

        # Debug prints
        print("Number of images with captions loaded:", len(self.image_filenames))
        print("DataLoader: Initialization complete.")

    def _load_and_map_flickr_captions(self, caption_file_path):
        """
        Loads captions from the Flickr 8k captions.txt file (CSV format)
        and maps them to image filenames.
        """
        img_to_captions = {}
        try:
            df = pd.read_csv(caption_file_path)
        except Exception as e:
            print(f"Error reading caption file {caption_file_path}: {e}")
            return {}

        # The Flickr8k captions.txt has 'image' and 'caption' columns
        # Example: 1000268201_693b08cb0e.jpg,A child in a pink dress is climbing up a set of stairs in an entry way .
        for index, row in df.iterrows():
            image_name = row['image']
            caption = row['caption']
            if image_name not in img_to_captions:
                img_to_captions[image_name] = []
            img_to_captions[image_name].append(caption)
        
        print(f"DataLoader: Loaded {len(img_to_captions)} unique images with captions.")
        # Debug: Print some sample image IDs and file names for the new format
        print("Sample images and captions:", list(img_to_captions.items())[:5])
        return img_to_captions

    # The following methods are now redundant or replaced by _load_and_map_flickr_captions
    # def load_captions(self, caption_file):
    #     """Original: Loads caption data from a JSON file."""
    #     pass # This function is no longer used

    # def map_image_ids_to_file_names(self, captions_data, image_dir):
    #     """Original: Maps image IDs to their corresponding file names."""
    #     pass # This function is no longer used

    # def load_captions_for_images(self):
    #     """Original: Organizes captions by image file name from COCO structure."""
    #     pass # This function is no longer used

    def build_vocabulary(self):
        """Builds a word-to-index and index-to-word vocabulary from the loaded captions."""
        all_words = []
        # Iterate over the values (lists of captions) in self.captions dictionary
        for caption_list in self.captions.values():
            for caption_text in caption_list:
                # Simple tokenization: lowercase and split by space
                all_words.extend(caption_text.lower().split())

        word_counts = Counter(all_words)
        
        # Initialize with special tokens
        vocab = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3} 
        idx = len(vocab) # Start index for new words after special tokens

        # Add words that meet the minimum frequency
        for word, count in word_counts.items():
            if count >= self.min_word_freq:
                vocab[word] = idx
                idx += 1

        word_to_idx = vocab
        idx_to_word = {idx: word for word, idx in vocab.items()}

        return {'word_to_idx': word_to_idx, 'idx_to_word': idx_to_word}

    def tokenize_and_encode_caption(self, caption_text, max_len=50): # Ensure max_len has a default
        """Tokenizes, encodes, and pads/truncates a single caption."""
        tokens = caption_text.lower().split()
        
        encoded_caption = [self.word_to_idx['<start>']]
        for token in tokens:
            encoded_caption.append(self.word_to_idx.get(token, self.word_to_idx['<unk>']))
        encoded_caption.append(self.word_to_idx['<end>'])

        # Pad or truncate to max_len
        if len(encoded_caption) > max_len:
            encoded_caption = encoded_caption[:max_len]
        while len(encoded_caption) < max_len:
            encoded_caption.append(self.word_to_idx['<pad>'])
        
        return np.array(encoded_caption, dtype=np.int64) # Use int64 for PyTorch LongTensor

    def load_image(self, image_file):
        """Loads and preprocesses an image for Flickr8k dataset."""
        # Flickr8k images are usually in a subdirectory like 'Flicker8k_Dataset'
        # Adjust this path based on where your actual images are relative to image_dir
        # Assuming image_dir is 'data/images' and actual images are in 'data/images/Flicker8k_Dataset'
        img_path = os.path.join(self.image_dir, 'Flicker8k_Dataset', image_file)
        
        if not os.path.exists(img_path):
            # Try loading directly from image_dir if not found in subdirectory
            img_path = os.path.join(self.image_dir, image_file)
            if not os.path.exists(img_path):
                print(f"Error: Image file not found at {img_path}. Skipping.")
                return None # Return None if image not found

        try:
            img = Image.open(img_path).convert('RGB') # Ensure RGB for consistent channels
            img = img.resize(self.image_size)
            img = np.array(img) / 127.5 - 1.0  # Normalize to [-1, 1]
            return img
        except Exception as e:
            print(f"Error loading image {image_file}: {e}. Skipping.")
            return None


    def get_batch(self, batch_size, max_caption_length=50):
        """Generates a batch of images and encoded captions."""
        batch_images = []
        batch_encoded_captions = []

        all_image_files_with_captions = self.image_filenames # Use the sorted list of filenames
        
        # Handle case where there are fewer images than batch_size
        if len(all_image_files_with_captions) == 0:
            print("Error: No unique images found with captions to form a batch.")
            return np.array([]), np.array([])

        if len(all_image_files_with_captions) < batch_size:
            # If dataset is smaller than batch_size, sample with replacement
            selected_image_files = np.random.choice(all_image_files_with_captions, batch_size, replace=True)
            print(f"Warning: Not enough unique images ({len(all_image_files_with_captions)}) for batch size {batch_size}. Sampling with replacement.")
        else:
            # Sample without replacement for full batches
            selected_image_files = np.random.choice(all_image_files_with_captions, batch_size, replace=False)

        for image_file in selected_image_files:
            img = self.load_image(image_file)
            if img is None: # Skip if image loading failed
                continue
            
            # Randomly choose one caption for the selected image
            raw_caption = random.choice(self.captions[image_file])
            
            encoded_caption = self.tokenize_and_encode_caption(raw_caption, max_len=max_caption_length)

            batch_images.append(img)
            batch_encoded_captions.append(encoded_caption)

        # Ensure we have enough data to form a batch
        if len(batch_images) == 0:
            print("Error: After filtering, no valid images or captions loaded to form a batch.")
            return np.array([]), np.array([]) # Return empty arrays

        # Convert lists to numpy arrays
        batch_images_np = np.array(batch_images)
        batch_encoded_captions_np = np.array(batch_encoded_captions)

        # Debug: Print the shape of the batch images and captions
        print("Batch images shape:", batch_images_np.shape)
        print("Batch encoded captions shape:", batch_encoded_captions_np.shape)

        return batch_images_np, batch_encoded_captions_np

# Example usage (remove or guard with if __name__ == "__main__": if this file is imported)
if __name__ == "__main__":
    # This example now assumes you have the Flickr8k dataset structure
    # image_dir points to the parent directory of 'Flicker8k_Dataset'
    # caption_file points directly to 'captions.txt'
    
    # Create dummy data for testing the Flickr8k format
    dummy_base_dir = 'data_dummy_flickr'
    dummy_image_dir = os.path.join(dummy_base_dir, 'images')
    dummy_flickr_dataset_dir = os.path.join(dummy_image_dir, 'Flicker8k_Dataset')
    dummy_caption_file = os.path.join(dummy_base_dir, 'captions.txt')
    
    os.makedirs(dummy_flickr_dataset_dir, exist_ok=True)
    os.makedirs(os.path.dirname(dummy_caption_file), exist_ok=True)

    # Create dummy images
    Image.new('RGB', (64, 64), color='red').save(os.path.join(dummy_flickr_dataset_dir, 'image1.jpg'))
    Image.new('RGB', (64, 64), color='green').save(os.path.join(dummy_flickr_dataset_dir, 'image2.jpg'))
    Image.new('RGB', (64, 64), color='blue').save(os.path.join(dummy_flickr_dataset_dir, 'image3.jpg'))

    # Create dummy captions.txt
    dummy_captions_content = """image,caption
image1.jpg,A red square.
image1.jpg,The color red.
image2.jpg,A green circle.
image3.jpg,A blue rectangle.
image3.jpg,The blue shape.
"""
    with open(dummy_caption_file, 'w') as f:
        f.write(dummy_captions_content)

    print("\n--- Running DataLoader Example Usage (Flickr8k format) ---")
    try:
        # Pass the directory containing 'Flicker8k_Dataset' and 'captions.txt'
        data_loader_example = DataLoader(dummy_image_dir, dummy_caption_file, min_word_freq=1)
        
        # Test get_batch
        batch_images_ex, batch_captions_ex = data_loader_example.get_batch(batch_size=2)
        print("Example batch images shape:", batch_images_ex.shape)
        print("Example batch captions shape:", batch_captions_ex.shape)

        # Test tokenize_and_encode_caption
        test_caption = "a cat on the roof"
        encoded = data_loader_example.tokenize_and_encode_caption(test_caption)
        print(f"Encoded '{test_caption}': {encoded}")

    except Exception as e:
        print(f"An error occurred during DataLoader example usage: {e}")
    finally:
        # Clean up dummy data
        print("--- Cleaning up dummy data ---")
        os.remove(os.path.join(dummy_flickr_dataset_dir, 'image1.jpg'))
        os.remove(os.path.join(dummy_flickr_dataset_dir, 'image2.jpg'))
        os.remove(os.path.join(dummy_flickr_dataset_dir, 'image3.jpg'))
        os.rmdir(dummy_flickr_dataset_dir)
        os.rmdir(dummy_image_dir) # Remove the 'images' subfolder
        os.remove(dummy_caption_file)
        os.rmdir(dummy_base_dir) # Remove the base dummy directory
        print("--- DataLoader Example Usage Finished ---")