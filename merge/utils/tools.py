import torch


def auto_device(device: str = None) -> torch.device:
    """
    Utility function to validate or determine a torch.device.

    Args:
        device (str, optional): The device string (e.g., "cuda", "cpu"). Defaults to None.

    Returns:
        torch.device: A validated or determined torch.device instance.
    """
    if device:
        try:
            validated_device = torch.device(device)
            if validated_device.type == "cuda" and not torch.cuda.is_available():
                raise ValueError(
                    f"CUDA is not available, but 'cuda' device was requested."
                )
            return validated_device
        except Exception as e:
            raise ValueError(f"Invalid device string provided: {device}. Error: {e}")
    else:
        # Determine an available device
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.has_mps:
            return torch.device("mps")
        else:
            return torch.device("cpu")
