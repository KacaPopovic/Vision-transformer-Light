import torch.nn as nn
from PIL import Image
from torchvision import transforms, datasets
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class LightViT(nn.Module):
    """
    LightViT is a PyTorch module that implements a lightweight version of the Vision Transformer (ViT) model for image
    classification tasks. It consists of an encoder composed of multiple ViTEncoder layers.

    Args:
        image_dim (tuple): The dimensions of the input images in the format (batch_size, channels, height, width).
        n_patches (int): The number of patches to divide the images into. Default is 7.
        n_blocks (int): The number of ViTEncoder blocks in the encoder. Default is 2.
        d (int): The dimensionality of the model's hidden state. Default is 8.
        n_heads (int): The number of attention heads in each ViTEncoder block. Default is 2.
        num_classes (int): The number of output classes for classification. Default is 10.

    Attributes:
        image_dim (tuple): The dimensions of the input images.
        n_patches (int): The number of patches.
        n_blocks (int): The number of ViTEncoder blocks.
        d (int): The dimensionality of the hidden state.
        n_heads (int): The number of attention heads.
        num_classes (int): The number of output classes.
        patches (None): A placeholder attribute for storing the extracted image patches.
        linear_map (CustomLinear): A custom linear layer for mapping the patches.
        cls_token (nn.Parameter): The learnable classification token.
        pos_embed (torch.Tensor): The positional embeddings for the patches.
        encoder_layers (nn.ModuleList): The list of ViTEncoder layers.
        classifier (nn.Sequential): The classifier module.

    Methods:
        forward(images): Performs forward pass of the LightViT model.
        get_patches(images, num_patches): Function for extracting image patches.
        get_pos_embeddings(num_patches, d): Generates the positional embeddings for the patches.
    """
    def __init__(self, image_dim, n_patches=7, n_blocks=2, d=8, n_heads=2, num_classes=10):
        super(LightViT, self).__init__()

        # Class Members
        self.image_dim = image_dim
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.d = d
        self.n_heads = n_heads
        self.num_classes = num_classes
        self.patches = None

        n, c, h, w = self.image_dim
        self.linear_map = CustomLinear(self.n_patches, h, w, c, self.d)  # P, H, W, C, d)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d))
        self.pos_embed = self.get_pos_embeddings(self.n_patches ** 2 + 1, self.d)
        self.encoder_layers = nn.ModuleList(
            [ViTEncoder(d_model=self.d, n_heads=self.n_heads) for _ in range(self.n_blocks)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d),
            nn.Linear(self.d, self.num_classes)
        )

    def forward(self, images):
        # Extract patches
        n, c, h, w = images.shape
        self.patches = self.get_patches(images, self.n_patches)
        x = self.linear_map(self.patches)

        # Add classification token
        cls_token = self.cls_token.expand(n, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add positional embeddings
        pos_embed = self.pos_embed.expand(n, -1, -1)
        x = x + pos_embed

        # Pass through encoder
        for layer in self.encoder_layers:
            x = layer(x)

        # Get classification token
        cls_token_final = x[:, 0]

        # Pass through classifier
        out = self.classifier(cls_token_final)

        return out

    def get_patches(self, images, num_patches):
        """
        Transfomers were initially created to process sequential data.
        In case of images, a sequence can be created through extracting patches.
        To do so, a crop window should be used with a defined window height and width.
        The dimension of data is originally in the format of (B,C,H,W),
        when transformed into patches and then flattened we get (B, PxP, (HxC/P)x(WxC/P)),
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
        # but the data for each image (the color, height, width) is flattened
        # into a single dimension–forming a long vector of pixel values.
        # we have n images, divided into num_patches*num_patches patches,
        # each of which has been flattened into  c * patch_h * patch_w dimensional vector
        patches = patches.view(n, -1, c * patch_h * patch_w)

        return patches

    def get_pos_embeddings(self, num_patches, d=8):
        positions = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2).float() * -(np.log(10000.0) / d))
        pos_embeddings = torch.empty(num_patches, d)
        pos_embeddings[:, 0::2] = torch.sin(positions * div_term)
        pos_embeddings[:, 1::2] = torch.cos(positions * div_term)
        pos_embeddings = pos_embeddings.unsqueeze(0)
        return pos_embeddings


class CustomLinear(nn.Module):
    """CustomLinear - Custom Linear Layer

    This class implements a custom linear layer that performs the forward pass of a linear transformation on the input tensor.

    Attributes:
        P (int): Number of patches.
        H (int): The height of the input tensor.
        W (int): The width of the input tensor.
        C (int): The number of channels in the input tensor.
        d (int): The output dimension of the linear layer - dimension of hidden representation - embedding.
        input_dim (int): The input feature dimension after flattening.
        linear (nn.Linear): The linear layer that performs the linear transformation.

    Methods:
        forward(x):
            Performs the forward pass of the linear transformation on the input tensor.

    """
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
    """
    ViTEncoder module that performs encoding using the Vision Transformer (ViT) architecture.

    Args:
        d_model (int): The number of features in the input and output (hidden size - embedding representation)
        n_heads (int): The number of attention heads.

    Attributes:
        hidden_d (int): The size of the hidden dimensions.
        n_heads (int): The number of attention heads.
        norm1 (nn.LayerNorm): Layer normalization 1.
        mhsa (MHSA): Multi-Head Self-Attention module.
        norm2 (nn.LayerNorm): Layer normalization 2.
        mlp (nn.Sequential): Multi-Layer Perceptron.

    Methods:
        forward(x): Performs the forward pass of the ViTEncoder module.

    Example:
        >>> model = ViTEncoder(d_model=512, n_heads=8)
        >>> input_tensor = torch.randn(1, 3, 224, 224)
        >>> output = model(input_tensor)
    """
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
    """MHSA Class

    Args:
        d (int): Dimension of embedding space.
        n_heads (int, optional): Dimension of attention heads. Defaults to 2.

    Attributes:
        n_heads (int): Dimension of attention heads.
        d (int): Dimension of embedding space.
        head_dim (int): Dimension of each head.

    Methods:
        forward(sequences): Performs forward pass of the MHSA module.

    """
    def __init__(self, d, n_heads=2):  # d: dimension of embedding spacr, n_head: dimension of attention heads
        super(MHSA, self).__init__()
        self.n_heads = n_heads
        self.d = d
        self.head_dim = d // n_heads

        assert self.head_dim * n_heads == d, "Embedding dimension must be divisible by n_heads"

        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.out = nn.Linear(d, d)

    def forward(self, sequences):
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
    """
    Load an image from the specified image_path and convert it to a tensor.

    :param image_path: The path to the image file.
    :return: The image as a tensor.
    """
    img_PIL = Image.open(image_path)
    img_tensor = transforms.ToTensor()(img_PIL)
    return img_tensor


def train(model, device, train_loader, optimizer, criterion, epoch):
    """
    Train the model for one epoch.

    :param model: The model to train.
    :param device: The device to use for training.
    :param train_loader: The data loader for training data.
    :param optimizer: The optimizer to use for training.
    :param criterion: The loss criterion to use.
    :param epoch: The current epoch.
    :return: The average loss per data point during training.
    """
    model.train()
    losses = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        losses += loss.item()
    return losses / len(train_loader) / train_loader.batch_size


def test(model, device, test_loader, criterion):
    """
    Calculate the test loss and accuracy for a given model.

    :param model: The model to be tested.
    :type model: torch.nn.Module
    :param device: The device used for computation (e.g., "cpu", "cuda").
    :type device: str
    :param test_loader: The data loader containing the test dataset.
    :type test_loader: torch.utils.data.DataLoader
    :param criterion: The loss function used for evaluating the model's output.
    :type criterion: torch.nn.modules.loss._Loss
    :return: A tuple containing the correct predictions and the average test loss.
    :rtype: tuple[int, float]
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')
    return correct, test_loss

def train_with_hyperparameters(learning_rates, batch_sizes):
    """
    :param learning_rates: List of learning rate values to be used during training.
    :param batch_sizes: List of batch size values to be used during training.
    :return: None

    This method trains a model using different combinations of learning rates and batch sizes. It uses the FashionMNIST dataset for training and testing. The best model with the highest accuracy is saved as "best_model.pth".

    The method follows these steps:

    1. Checks if GPU is available and sets the device accordingly.
    2. Defines the data transformations for the dataset.
    3. Loads the FashionMNIST dataset for training and testing.
    4. Initializes the best accuracy as 0.0 and the best model state as None.
    5. Loops over each learning rate in the given learning_rates list and each batch size in the given batch_sizes list.
    6. Sets up the data loaders for training and testing using the current batch size.
    7. Initializes the model, optimizer, criterion, train losses, test losses, and accuracy.
    8. Trains the model for 5 epochs, calculating the train loss and test loss for each epoch.
    9. Calculates the current accuracy and saves the model state if the accuracy is higher than the best accuracy seen so far.
    10. Appends the accuracy, train loss, and test loss to their respective lists.
    11. Plots the training and test losses.
    12. Prints the best model accuracy.
    13. Saves the best model state as "best_model.pth".
    """
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Define Dataloader
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    fashion_mnist_train = datasets.FashionMNIST(root='.', train=True, transform=transform, download=True)
    fashion_mnist_test = datasets.FashionMNIST(root='.', train=False, transform=transform, download=True)

    # Initialize best accuracy as zero
    best_acc = 0.0
    best_model_state = None

    for lr in learning_rates:
        for bs in batch_sizes:
            train_loader = DataLoader(fashion_mnist_train, batch_size=bs, shuffle=True)
            test_loader = DataLoader(fashion_mnist_test, batch_size=bs, shuffle=False)
            model = LightViT(image_dim=(32, 1, 28, 28)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            train_losses = []
            test_losses = []
            accuracy = []
            # Train
            for epoch in range(1, 6):
                train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
                correct, test_loss = test(model, device, test_loader, criterion)
                current_accuracy = 100. * correct / len(test_loader.dataset)
                # Save this model if it has better accuracy
                if current_accuracy > best_acc:
                    best_acc = current_accuracy
                    best_model_state = model.state_dict()
                accuracy.append(current_accuracy)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
            # Plot training loss
            epochs = range(1, len(train_losses) + 1)
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, train_losses, label='Training loss')
            plt.plot(epochs, test_losses, label='Test loss')
            plt.title(f'Loss for LR={lr}, BS={bs} with Final Accuracy: {accuracy[-1]:.2f}%')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
    print(f"Best Model Accuracy: {best_acc}%")
    # Save the best model
    torch.save(best_model_state, "best_model.pth")

def main():
    """
    This is the main method for training and evaluating the LightViT model on the MNIST dataset.

    :return: None
    """
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define Dataloader
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_train = datasets.MNIST(root='.', train=True, transform=transform, download=True)
    mnist_test = datasets.MNIST(root='.', train=False, transform=transform, download=True)

    train_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)

    model = LightViT(image_dim=(32, 1, 28, 28)).to(device)
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_parameters = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'The model has {trainable_parameters} trainable parameters')
    print(f'The model has {non_trainable_parameters} non-trainable parameters')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_losses = []
    accuracy = []

    # Train
    for epoch in range(1, 6):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        correct, test_loss = test(model, device, test_loader, criterion)
        accuracy.append(100. * correct / len(test_loader.dataset))
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    # Plot training loss
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training loss')
    plt.plot(epochs, test_losses, label='Test loss')
    plt.title(f'Loss with Final Accuracy: {accuracy[-1]:.2f}%')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [32, 64, 128]
    train_with_hyperparameters(learning_rates, batch_sizes)
