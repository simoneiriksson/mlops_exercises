import torch
import click


def predict(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([model(batch) for batch in dataloader], 0)


@click.command()
@click.argument("model_checkpoint", default="models/trained_model.pt")
@click.argument("input_images", default="data/processed/test_images.pt")
def predict_model(model_checkpoint, input_images):
    model = torch.load(model_checkpoint)  # load the model
    images = torch.load(input_images)  # laod the images
    images = (images - images.mean(dim=[1, 2, 3], keepdim=True)) / images.std(
        dim=[1, 2, 3], keepdim=True
    )  # normalize the images
    predictions = predict(model, images)  # run prediction
    print(f"{predictions.shape = }")  # print the shape of the predictions
    torch.save(predictions, "data/predictions.pt")
    print("Predictions saved to data/predictions.pt")
    print(f"{predictions = }")


# cli.add_command(predict_model)

if __name__ == "__main__":
    predict_model()
