import torch

def create_frustum(image_size, feature_size):
    iH, iW = image_size
    fH, fW = feature_size

    ds = (
        torch.arange(*self.dbound, dtype=torch.float)
        .view(-1, 1, 1)
        .expand(-1, fH, fW) # 118， 32， 88
    )
    D, _, _ = ds.shape

    xs = (
        torch.linspace(0, iW - 1, fW, dtype=torch.float)
        .view(1, 1, fW)
        .expand(D, fH, fW)
    )
    ys = (
        torch.linspace(0, iH - 1, fH, dtype=torch.float)
        .view(1, fH, 1)
        .expand(D, fH, fW)
    )

    frustum = torch.stack((xs, ys, ds), -1)
    return frustum

if __name__ == "__main__":
    image_size = 256, 704  # H , W
    feature_size = 32, 88   # H , W
    create_frustum(image_size, feature_size)