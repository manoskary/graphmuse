import torch


def compare_all_elements(u, v, max_val, data_split=1):
    """
    Description.....

    Parameters
    ----------
        u:         first array to be compared (1D torch.tensor of ints)
        v:         second array to be compared (1D torch.tensor of ints)
        max_val:         the largest element in either tensorA or tensorB (real number)
        data_split:      the number of subsets to split the mask up into (int)
    Returns
    -------
        compared_inds_a:  indices of tensorA that match elements in tensorB (1D torch.tensor of ints, type torch.long)
        compared_inds_b:  indices of tensorB that match elements in tensorA (1D torch.tensor of ints, type torch.long)
    """
    compared_inds_a, compared_inds_b, inc = torch.tensor([]).to(u.device), torch.tensor([]).to(u.device), int(
        max_val // data_split) + 1
    for iii in range(data_split):
        inds_a, inds_b = (iii * inc <= u) * (u < (iii + 1) * inc), (iii * inc <= v) * (
                v < (iii + 1) * inc)
        tile_a, tile_b = u[inds_a], v[inds_b]
        tile_a, tile_b = tile_a.unsqueeze(0).repeat(tile_b.size(0), 1), torch.transpose(tile_b.unsqueeze(0), 0, 1).repeat(1,
                                                                                                                     tile_a.size(
                                                                                                                         0))
        nz_inds = torch.nonzero(tile_a == tile_b, as_tuple=False)
        nz_inds_a, nz_inds_b = nz_inds[:, 1], nz_inds[:, 0]
        compared_inds_a, compared_inds_b = torch.cat((compared_inds_a, inds_a.nonzero()[nz_inds_a]), 0), torch.cat(
            (compared_inds_b, inds_b.nonzero()[nz_inds_b]), 0)
    return compared_inds_a.squeeze().long(), compared_inds_b.squeeze().long()