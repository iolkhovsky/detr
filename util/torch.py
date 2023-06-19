def tensor2cpu(tensor):
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.cpu()


def tensor2numpy(tensor):
    return tensor2cpu(tensor).numpy()
