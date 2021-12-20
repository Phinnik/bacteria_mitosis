import torch
from torchvision.models import vgg11

encoder = vgg11(pretrained=True).features


class SiameseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = torch.nn.Sequential(
            encoder,
            torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Flatten()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(8192 * 2, 512),
            torch.nn.Linear(512, 512),
            torch.nn.Linear(512, 1),
        )

    def encode_image(self, image):
        return self.image_encoder(image)

    def predict(self, encoding_1, encoding_2):
        X = torch.concat([encoding_1, encoding_2], dim=1)
        return self.decoder(X)

    def forward(self, image_1, image_2):
        encoding_1 = self.encode_image(image_1)
        encoding_2 = self.encode_image(image_2)
        return self.predict(encoding_1, encoding_2)

