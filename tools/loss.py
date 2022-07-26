import torch


def sequence_loss(est_flow, batch, gamma=0.8):
    n_predictions = len(est_flow)
    flow_loss = 0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        # part loss
        # i_loss = compute_loss(est_flow[i], batch)
        # full loss
        i_loss = compute_full_loss(est_flow[i], batch)
        flow_loss += i_weight * i_loss

    return flow_loss


def compute_loss(est_flow, batch):
    """
    Compute training loss.

    Parameters
    ----------
    est_flow : torch.Tensor
        Estimated flow.
    batch : flot.datasets.generic.Batch
        Contains ground truth flow and mask.

    Returns
    -------
    loss : torch.Tensor
        Training loss for current batch.

    """

    mask = batch["ground_truth"][0][..., 0]
    true_flow = batch["ground_truth"][1]
    error = est_flow - true_flow
    error = error[mask > 0]
    loss = torch.mean(torch.abs(error))
    # loss = torch.norm(error, dim=2).sum().mean()

    return loss

def compute_full_loss(est_flow, batch, alpha=1):
    """
        compute the full loss with the occlution point
    """
    mask = batch["ground_truth"][0][..., 0]
    true_flow = batch["ground_truth"][1]
    error = est_flow - true_flow
    error = error[mask >= 0]
    occ_loss = torch.mean(torch.abs(error))
    # occ_loss = torch.norm(error, dim=2).sum().mean()
    non_occ_loss = compute_loss(est_flow, batch)
    loss = non_occ_loss + alpha*occ_loss
    
    return loss
    
    