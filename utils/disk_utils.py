import torch
import sys


def load_model_state(seg_net, checkpoint):
    """ Load model weights from checkpoint into seg_net. """

    seg_net.encoder.load_state_dict(checkpoint['encoder'])
    seg_net.decoder.load_state_dict(checkpoint['decoder'])
    
    try:
        seg_net.projection_net.load_state_dict(checkpoint['projection_net'])
    except KeyError:
        print("No projection net weights found in model checkpoint.")
    except AttributeError:
        print("No projection net found in seg net.")

def load_checkpoint_if_exists(model, save_path):
    """
    If the save_path exists:
        load the model state from the checkpoint.
    Else:
        print an error message and exit the program.

    """
    try:
        checkpoint = torch.load(save_path, map_location=model.device)
        load_model_state(model, checkpoint)
    except FileNotFoundError:
        print(f"Checkpoint file not found: {save_path}")
        sys.exit()


def get_encoder_state_dict(model):
    """
    Returns the state dictionary of the encoder.
    Accounts for the use of LoRA.
    """
    if model.seg_net.encoder.lora_rank is not None:
        import loralib as lora
        return lora.lora_state_dict(model.seg_net.encoder)
    else:
        return model.seg_net.encoder.state_dict()
