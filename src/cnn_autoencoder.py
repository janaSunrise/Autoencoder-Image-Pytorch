import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
)
transform = transforms.ToTensor()

mnist_data = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
data_loader = torch.utils.data.DataLoader(
    dataset=mnist_data, batch_size=64, shuffle=True
)

# Generate the batches of Image and labels
dataiter = iter(data_loader)
images, labels = dataiter.next()

# Get the minimum and maximum using torch for the images.
print(torch.min(images), torch.max(images))


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Note: For `nn.MaxPool2d` use `nn.MaxUnpool2d`, or use different kernelsize, stride and such.
# For `Input [-1, +1]` use `nn.Tanh`

# Training the model!
model = Autoencoder()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

num_epochs = 30
outputs = []

for epoch in range(num_epochs):
    for img, _ in data_loader:
        recon = model(img)
        loss = criterion(recon, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch:{epoch+1} | Loss:{loss.item():.4f}")
    outputs.append((epoch, img, recon))

for k in range(0, num_epochs, 4):
    plt.figure(figsize=(9, 2))
    plt.gray()

    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()

    for i, item in enumerate(imgs):
        if i >= 9:
            break

        plt.subplot(2, 9, i + 1)
        plt.imshow(item[0])

    for i, item in enumerate(recon):
        if i >= 9:
            break

        plt.subplot(2, 9, 9 + i + 1)
        plt.imshow(item[0])

# Save the model
torch.save(model.state_dict(), "cnn_autoencoder.bin")
