import torch


class SiameseNet(torch.nn.Module):
    def __init__(self, in_height: int, in_width: int, in_channels: int):
        super().__init__()
        self.image_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 1, kernel_size=(3, 3)),
            torch.nn.Flatten()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear((in_height - 2) * (in_width - 2) * 2, 1)
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


if __name__ == '__main__':
    model = SiameseNet(512, 512, 3)
    img = torch.randn((3, 512, 512))
    out = model(img[None, ...], img[None, ...])
    print(out.shape)
