import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, codebook_size, embedding_dim):
        super(Encoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(1, stride=1),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(1, stride=1),
        )
        # Calculate the output size of the conv layers for a dummy input
        dummy_input = torch.zeros(1, 3, 10, 64, 64)
        dummy_output = self.conv_layers(dummy_input)
        conv_output_size = dummy_output.view(1, -1).size(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, codebook_size * embedding_dim),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class Decoder(nn.Module):
    def __init__(self, codebook_size, embedding_dim):
        super(Decoder, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(codebook_size * embedding_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 8 * 8 * 8),
            nn.ReLU(),
        )
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 3, kernel_size=2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc_layers(x)
        x = x.view(x.size(0), 128, 8, 8, 8)
        x = self.conv_layers(x)
        return x


class Quantizer(nn.Module):
    def __init__(self, codebook_size, embedding_dim):
        super(Quantizer, self).__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.codebook = nn.Embedding(codebook_size, embedding_dim)

    def forward(self, z):
        z = z.view(z.shape[0], self.embedding_dim, -1).permute(0, 2, 1)
        distances = (
            (z**2).sum(dim=-1, keepdim=True)
            + self.codebook.weight.pow(2).sum(dim=1)
            - 2 * torch.matmul(z, self.codebook.weight.t())
        )
        indices = distances.argmin(dim=-1)
        x = self.codebook(indices)
        return x, indices


class VisualTokenizer(nn.Module):
    def __init__(self, codebook_size, embedding_dim):
        super(VisualTokenizer, self).__init__()
        self.encoder = Encoder(codebook_size, embedding_dim)
        self.decoder = Decoder(codebook_size, embedding_dim)
        self.quantizer = Quantizer(codebook_size, embedding_dim)

    def forward(self, x):
        z = self.encoder(x)
        x, indices = self.quantizer(z)
        x = self.decoder(x)
        return x, indices


# Define the size of the codebook and the embedding dimension
codebook_size = 512
embedding_dim = 256

# Initialize the visual tokenizer
visual_tokenizer = VisualTokenizer(codebook_size, embedding_dim)

# Assume we have a batch of videos, each of size (T, H, W, 3)
# where T is the number of frames, H is the height, W is the width, and 3 is the number of color channels
videos = torch.randn(16, 3, 10, 64, 64)  # batch of 16 videos

# Pass the videos through the visual tokenizer
reconstructed_videos, indices = visual_tokenizer(videos)

# The reconstructed_videos tensor contains the reconstructed videos
# The indices tensor contains the indices of the tokens in the codebook
