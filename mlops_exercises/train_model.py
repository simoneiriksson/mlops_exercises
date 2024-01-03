# Testing testbranch

import click
import torch
from matplotlib import pyplot as plt
from torch import nn

# from data import mnist
from models.model import MyAwesomeModel


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--outfile", default="models/trained_model.pt", help="Output file name")
def train(lr, outfile):
    torch.random.manual_seed(0)
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = "cpu"
    print(f"{device = }")
    # TODO: Implement training loop here
    model = MyAwesomeModel().to(device)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.load("data/processed/train_images.pt"), torch.load("data/processed/train_labels.pt")
        ),
        batch_size=64,
        shuffle=True,
    )

    epochs = 10
    train_losses = []
    for epoch in range(epochs):
        loss_accum = 0  # to keep track of the loss value
        for images, labels in train_loader:
            # images = images.view(images.shape[0], -1).to(device)
            images = images.to(device)
            # print(f"{images.shape = }")
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()
            loss_accum += loss.item()
        train_losses.append(loss_accum / len(train_loader))
        print(
            "Epoch: {}/{}.. ".format(epoch + 1, epochs),
            "Training Loss: {:.3f}.. ".format(loss_accum / len(train_loader)),
        )
    plt.plot(train_losses)
    plt.savefig("reports/figures/train_losses.png")
    torch.save(model, outfile)


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    testloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.load("data/processed/test_images.pt"), torch.load("data/processed/test_labels.pt")
        ),
        batch_size=64,
        shuffle=True,
    )

    with torch.no_grad():
        model.eval()
        num_equals = 0
        num_tot = 0
        for images, labels in testloader:
            # Get the class probabilities
            ps = torch.exp(model(images))
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            # print(f"{equals.shape = }")
            num_equals += equals.sum()
            num_tot += equals.shape[0]
            # print(f"{num_equals=}, {num_tot=}")
        ## TODO: Implement the validation pass and print out the validation accuracy
        accuracy = num_equals / num_tot
        print(f"Accuracy: {accuracy.item():0.2%}")


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
