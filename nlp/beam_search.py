import torch


@torch.no_grad()
def beam_search(model, x, beam_width, output_length, block_size=None, exclude_classes=None, temperature=1):
    """
    Beam search for PyTorch.
    See `Beam search <https://en.wikipedia.org/wiki/Beam_search>`_.

    Args:
        model (Module): The model to predict output classes. It is expected to output unnormalized logits.
        x (Tensor): Input class indices (1-d, int64).
        beam_width (int): The number of branches (output combinations) to track while searching.
        output_length (int): The number of predictions to generate.
        block_size (int): The maximum sequence length that the model accepts. ``None`` denotes unlimited.
        exclude_classes (list[int]): Indices of classes to exclude from results (e.g. padding and unknown).
        temperature (float): A divisor for the logits to flatten (if < 1) or emphasize (if > 1) class probabilities.

    Returns:
        Tensor of output class indices (1-d, int64).
    """
    model.eval()
    empty = torch.tensor([], dtype=torch.int64, device=model.device)
    root = (empty, 1.0)
    branches = [root]
    for _ in range(output_length):
        candidates = []
        for branch_path, branch_proba in branches:
            _x, _path = x, branch_path
            if block_size:
                _path = branch_path[-block_size:]
                _x = x[max(x.size(0) + _path.size(0) - block_size, 0):]
            inputs = torch.cat([_x, _path]).unsqueeze(0)
            logits = model(inputs).squeeze(0)[-1]
            if exclude_classes:
                logits[exclude_classes] = float('-inf')
            logits = logits / temperature
            probas = logits.softmax(0)
            probas, indices = probas.topk(beam_width)
            probas *= branch_proba
            cand = [(torch.cat([branch_path, indices[i:i+1]]), proba) for i, proba in enumerate(probas)]
            candidates.extend(cand)
        candidates = sorted(candidates, key=lambda c: c[1], reverse=True)
        branches = candidates[:beam_width]
    return branches[0][0]
