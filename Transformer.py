import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import torch.nn.functional as F


class LightViT(nn.Module):

    def __init__(self, image_dim, n_patches=7, n_blocks=2, d=8, n_heads=2, num_classes=10):
        super(LightViT, self).__init__()

        ## Class Members
        self.image_dim = image_dim
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.d = d
        self.n_heads = n_heads
        self.num_classes = num_classes
        self.patches = None

        ## 1B) Linear Mapping
        n, c, h, w = self.image_dim
        self.linear_map = CustomLinear(self.n_patches, h, w, c, self.d) #P, H, W, C, d)
        ## 2A) Learnable Parameter
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d))
        ## 2B) Positional embedding
        self.pos_embed = self.get_pos_embeddings(self.n_patches**2 + 1, self.d)
        ## 3) Encoder blocks
        self.encoder = ViTEncoder(self.d, self.n_heads)

        # 5) Classification Head
        self.classifier = None

    def forward(self, images):
        ## Extract patches
        n, c, h, w = self.image_dim
        self.patches = self.get_patches(images, self.n_patches)
        x = self.linear_map(self.patches)

        ## Add classification token
        cls_token = self.cls_token
        x = torch.cat((cls_token, x), dim=1)

        ## Add positional embeddings
        pos_embed = self.pos_embed
        x = x + pos_embed

        ## Pass through encoder
        x = self.encoder(x)

        ## Get classification token

        ## Pass through classifier

        return x


    def get_patches(self, images, num_patches):
        """
        Transfomers were initially created to process sequential data.
        In case of images, a sequence can be created through extracting patches.
        To do so, a crop window should be used with a defined window height and width.
        The dimension of data is originally in the format of (B,C,H,W),
        when transorfmed into patches and then flattened we get (B, PxP, (HxC/P)x(WxC/P)),
        where B is the batch size and PxP is total number of patches in an image.
        In this example, you can set P=7.
        Output: A function that extracts image patches.
        The output format should have a shape of (B,49,16).
        The function will be used inside LightViT class.
        """
        if len(images.shape) == 3:
                images = images.unsqueeze(0)

        # Image Dimensions
        n, c, h, w = images.shape
        # Size of each patch
        patch_h = patch_w = h // num_patches

        # Generate patches
        patches = images.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)

        # Re-arrange the patches to satisfy our required dimensions (B, P*P, CxHxW/P*P)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()

        # Flatten the last dimensions - The batch size, remains the same,
        # but the data for each image (the color, height, width) is flattened into a single dimension–forming a long vector of pixel values.
        # we have n images, devided into num_patches*num_patches patches, each of which has been flattened into  c * patch_h * patch_w dimensional vector
        patches = patches.view(n, -1, c * patch_h * patch_w)

        return patches

    def get_pos_embeddings(self, num_patches, d=8):
        positions = torch.arange(0,num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2).float() * -(np.log(10000.0) / d))
        pos_embeddings = torch.empty(num_patches, d)
        pos_embeddings[:, 0::2] = torch.sin(positions * div_term)
        pos_embeddings[:, 1::2] = torch.cos(positions * div_term)
        pos_embeddings = pos_embeddings.unsqueeze(0).transpose(0, 1)
        return pos_embeddings


class CustomLinear(nn.Module):
    def __init__(self, P, H, W, C, d):
        super(CustomLinear, self).__init__()
        self.P = P
        self.H = H
        self.W = W
        self.C = C
        self.d = d

        # The input feature dimension after flattening
        self.input_dim = (H * C // P) * (W * C // P)

        # Define the linear layer
        self.linear = nn.Linear(self.input_dim, d)

    def forward(self, x):
        # x shape: (B, P * P, (H * C / P) * (W * C / P))
        B, _, _ = x.shape

        # Flatten the last dimension
        x = x.view(B, self.P * self.P, self.input_dim)

        # Apply the linear layer
        x = self.linear(x)

        return x

class ViTEncoder(nn.Module):
    def __init__(self, d_model, n_heads):
        super(ViTEncoder, self).__init__()
        self.hidden_d = d_model
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.mhsa = MHSA(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        attn_out = self.mhsa(self.norm1(x))
        x = x + attn_out
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out

        return x

class MHSA(nn.Module):
    def __init__(self, d, n_heads=2): # d: dimension of embedding spacr, n_head: dimension of attention heads
        super(MHSA, self).__init__()
        self.n_heads = n_heads
        self.d = d
        self.head_dim = d // n_heads

        assert self.head_dim * n_heads == d, "Embedding dimension must be divisible by n_heads"

        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def forward(self,sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # Shape is transformed to   (N, seq_length, n_heads, token_dim / n_heads)
        # And finally we return back    (N, seq_length, item_dim)  (through concatenation)
        N, seq_length, token_dim = sequences.shape

        # Linear projections
        Q = self.query(sequences)
        K = self.key(sequences)
        V = self.value(sequences)

        # Split into multiple heads
        Q = Q.view(N, seq_length, self.n_heads, self.head_dim)
        K = K.view(N, seq_length, self.n_heads, self.head_dim)
        V = V.view(N, seq_length, self.n_heads, self.head_dim)

        # Permute to get shape (N, n_heads, seq_length, head_dim)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn, V)

        # Concatenate heads
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(N, seq_length, self.d)

        # Final linear layer
        output = self.out(attn_output)

        return output



def load_image(image_path):
    img_PIL = Image.open(image_path)
    img_tensor = transforms.ToTensor()(img_PIL)
    return img_tensor

def main():
    # Load image
    img_tensor = load_image('img_1.jpg')
    images = img_tensor
    images = images.unsqueeze(0)
    model = LightViT(image_dim=images.shape)
    output = model(images)
    K=50



if __name__ == '__main__':
    main()