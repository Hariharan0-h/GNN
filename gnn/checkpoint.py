import torch


def load_model_checkpoint(model, path):

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    return model