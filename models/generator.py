# e:\hello\models\generator.py (Corrected Version)

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, noise_dim, vocab_size, embedding_dim, caption_encoder_hidden_size, output_channels=3):
        super(Generator, self).__init__()

        self.noise_dim = noise_dim
        self.embedding_dim = embedding_dim
        self.caption_encoder_hidden_size = caption_encoder_hidden_size
        self.output_channels = output_channels

        # 1. Word Embedding Layer: Converts word IDs to dense vectors
        # This will take in numerical caption IDs and output (batch_size, seq_len, embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 2. Text Encoder: Processes the sequence of embeddings into a fixed-size vector
        # For simplicity, we'll use a linear layer on the mean of the embeddings.
        # More advanced: Use an RNN (GRU/LSTM) to capture sequential information.
        self.text_encoder = nn.Sequential(
            nn.Linear(embedding_dim, caption_encoder_hidden_size),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Calculate the total input dimension for the generator's main model
        # This is noise_dim + the output dimension of your text_encoder
        self.total_input_dim = self.noise_dim + self.caption_encoder_hidden_size

        # 3. Main Generator Model (from the concatenated noise and caption embedding)
        self.model = nn.Sequential(
            # Foundation for 4x4 image: input is total_input_dim
            nn.Linear(self.total_input_dim, 256 * 4 * 4, bias=False),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # Reshape to (batch_size, 256, 4, 4)
            nn.Unflatten(1, (256, 4, 4)),

            # Upsample to 8x8
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Upsample to 16x16
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Upsample to 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Upsample to 64x64
            nn.ConvTranspose2d(32, self.output_channels, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.Tanh() # Output images in [-1, 1] range
        )

    def forward(self, z, captions_indexed): # Generator now accepts noise AND indexed captions
        # Process captions:
        # 1. Embed the word IDs
        embedded_captions = self.embedding(captions_indexed) # (batch_size, seq_len, embedding_dim)
        
        # 2. Encode the sequence of embeddings into a single fixed-size vector
        # Simple approach: mean pooling across the sequence dimension
        caption_features = embedded_captions.mean(dim=1) # (batch_size, embedding_dim)
        caption_features = self.text_encoder(caption_features) # (batch_size, caption_encoder_hidden_size)

        # Concatenate noise vector with caption features
        combined_input = torch.cat([z, caption_features], dim=1) # (batch_size, noise_dim + caption_encoder_hidden_size)

        # Pass the combined input to the main generator model
        return self.model(combined_input)