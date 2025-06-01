# e:\hello\models\discriminator.py (Corrected Version)

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, image_shape, vocab_size, embedding_dim, caption_encoder_hidden_size):
        super(Discriminator, self).__init__()

        self.image_shape = image_shape # (channels, height, width)
        self.embedding_dim = embedding_dim
        self.caption_encoder_hidden_size = caption_encoder_hidden_size # This will be the output size of processed text

        # 1. Image processing layers (unchanged from your code)
        self.conv1 = nn.Conv2d(image_shape[0], 64, kernel_size=4, stride=2, padding=1) # 64x64 -> 32x32
        self.leaky_relu1 = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # 32x32 -> 16x16
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.leaky_relu2 = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(0.3)

        # Calculate the flattened size of the image features after conv layers
        # For 64x64 input: 128 channels * (64/4) * (64/4) = 128 * 16 * 16 = 32768
        self.flattened_image_features = 128 * (image_shape[1] // 4) * (image_shape[2] // 4)

        # 2. Text processing layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # Converts word IDs to embeddings
        self.text_encoder = nn.Sequential( # Encodes sequence of embeddings into fixed size
            nn.Linear(embedding_dim, self.caption_encoder_hidden_size), # maps embedding_dim to desired hidden_size
            nn.LeakyReLU(0.2)
        )

        # 3. Combined layers
        # The input to combined_dense will be flattened_image_features + caption_encoder_hidden_size
        self.combined_dense = nn.Linear(self.flattened_image_features + self.caption_encoder_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image, captions_indexed): # Discriminator now takes image AND indexed captions
        # Process image
        x = self.conv1(image)
        x = self.leaky_relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x)

        x = torch.flatten(x, 1) # Flatten image features: (batch_size, self.flattened_image_features)

        # Process text
        # 1. Embed the word IDs
        embedded_captions = self.embedding(captions_indexed) # (batch_size, seq_len, embedding_dim)
        
        # 2. Encode the sequence of embeddings into a single fixed-size vector
        # Simple approach: mean pooling across the sequence dimension
        y = embedded_captions.mean(dim=1) # (batch_size, embedding_dim)
        y = self.text_encoder(y) # (batch_size, caption_encoder_hidden_size)

        # Combine features
        combined = torch.cat([x, y], dim=1) # Concatenate image and text features

        # Final output
        z = self.combined_dense(combined)
        z = self.sigmoid(z)

        return z