import torch


@torch.no_grad()
def sample(model, x, output_length, block_size, exclude_classes=None, temperature=1, top_k=None, top_p=None):
    """
    Sampling for autoregressive models in PyTorch.
    See `How to generate text <https://huggingface.co/blog/how-to-generate>`_.

    Args:
        model (Module): The model to predict output classes. It is expected to output unnormalized logits.
        x (Tensor): Input class indices (1-d, int64).
        output_length (int): The number of predictions to generate.
        block_size (int): The maximum sequence length that the model accepts. ``None`` denotes unlimited.
        exclude_classes (list[int]): Indices of classes to exclude from results (e.g. padding and unknown).
        temperature (float): A divisor for the logits to flatten (if < 1) or emphasize (if > 1) class probabilities.
        top_k (float): The number of most likely classes, from which to sample each next class. Set to ``1`` for greedy search.
        top_p (float): The minimum probability of the set of classes to sample from (aka "nucleus sampling" or "dynamic top-k").
            Can be combined with ``top_k``.

    Returns:
        Tensor of output class indices (1-d, int64).
    """
    model.eval()
    seq = x
    for _ in range(output_length):
        inputs = seq[-block_size:].unsqueeze(0)
        logits = model.forward(inputs).squeeze(0)[-1]
        if exclude_classes:
            logits[exclude_classes] = float('-inf')
        logits = logits / temperature
        probas = logits.softmax(dim=-1)
        if top_k:
            probas, indices = probas.topk(top_k)
        else:
            indices = torch.arange(probas.size(-1))
        if top_p:
            sorted_probas, sorted_indices = probas.sort()  # ascending sort simplifies the following
            cumprobas = sorted_probas.cumsum(-1)
            nucleus_size = cumprobas.size(-1) - torch.sum(cumprobas <= (1 - top_p))
            nucleus_indices = sorted_indices[-nucleus_size:]
            probas = sorted_probas[-nucleus_size:]
            indices = indices[nucleus_indices]
        index = probas.multinomial(1)
        seq = torch.cat([seq, indices[index]])
    return seq[x.size(-1):]
