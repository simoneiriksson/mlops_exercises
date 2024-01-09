import torch


def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    train_images, train_labels = [], []
    for i in range(10):
        train_images.append(torch.load("data/raw/train_images_{}.pt".format(i)))
        train_labels.append(torch.load("data/raw/train_target_{}.pt".format(i)))
    train_images = torch.cat(train_images)
    train_labels = torch.cat(train_labels)

    test_images = torch.load("data/raw/test_images.pt")
    test_labels = torch.load("data/raw/test_target.pt")

    # print(f"{train_images.shape = }")
    train_images = train_images.unsqueeze(1)
    test_images = test_images.unsqueeze(1)
    # print(f"{train_images.shape = }")

    # normalize the data
    train_images = (train_images - train_images.mean(dim=[1, 2, 3], keepdim=True)) / train_images.std(
        dim=[1, 2, 3], keepdim=True
    )
    test_images = (test_images - test_images.mean(dim=[1, 2, 3], keepdim=True)) / test_images.std(
        dim=[1, 2, 3], keepdim=True
    )

    torch.save(train_images, "data/processed/train_images.pt")
    torch.save(train_labels, "data/processed/train_labels.pt")
    torch.save(test_images, "data/processed/test_images.pt")
    torch.save(test_labels, "data/processed/test_labels.pt")


if __name__ == "__main__":
    # Get the data and process it
    mnist()
