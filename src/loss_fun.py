import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor


class PermutationLoss(nn.Module):
    r"""
    Binary cross entropy loss between two permutations, also known as "permutation loss".
    Proposed by `"Wang et al. Learning Combinatorial Embedding Networks for Deep Graph Matching. ICCV 2019."
    <http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Learning_Combinatorial_Embedding_Networks_for_Deep_Graph_Matching_ICCV_2019_paper.pdf>`_

    .. math::
        L_{perm} =- \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2}
        \left(\mathbf{X}^{gt}_{i,j} \log \mathbf{S}_{i,j} + (1-\mathbf{X}^{gt}_{i,j}) \log (1-\mathbf{S}_{i,j}) \right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    .. note::
        For batched input, this loss function computes the averaged loss among all instances in the batch.
    """
    def __init__(self):
        super(PermutationLoss, self).__init__()

    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor) -> Tensor:
        r"""
        :param pred_dsmat: :math:`(b\times n_1 \times n_2)` predicted doubly-stochastic matrix :math:`(\mathbf{S})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :param src_ns: :math:`(b)` number of exact pairs in the first graph (also known as source graph).
        :param tgt_ns: :math:`(b)` number of exact pairs in the second graph (also known as target graph).
        :return: :math:`(1)` averaged permutation loss

        .. note::
            We support batched instances with different number of nodes, therefore ``src_ns`` and ``tgt_ns`` are
            required to specify the exact number of nodes of each instance in the batch.
        """
        batch_num = pred_dsmat.shape[0]

        pred_dsmat = pred_dsmat.to(dtype=torch.float32)

        try:
            assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
            assert torch.all((gt_perm >= 0) * (gt_perm <= 1))
        except AssertionError as err:
            print(pred_dsmat)
            raise err

        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            batch_slice = [b]
            loss += F.binary_cross_entropy(
                pred_dsmat[batch_slice],
                gt_perm[batch_slice],
                reduction='sum')
            n_sum += 1

        return loss / n_sum

class HammingLoss(nn.Module):
    r"""
    Hamming loss between two permutations.

    .. math::
        L_{hamm} = \sum_{i \in \mathcal{V}_1, j \in \mathcal{V}_2}
        \left(\mathbf{X}_{i,j} (1-\mathbf{X}^{gt}_{i,j}) +  (1-\mathbf{X}_{i,j}) \mathbf{X}^{gt}_{i,j}\right)

    where :math:`\mathcal{V}_1, \mathcal{V}_2` are vertex sets for two graphs.

    Firstly adopted by `"Rolinek et al. Deep Graph Matching via Blackbox Differentiation of Combinatorial Solvers.
    ECCV 2020." <https://arxiv.org/abs/2003.11657>`_

    .. note::
        Hamming loss is defined between two discrete matrices, and discretization will in general truncate gradient. A
        workaround may be using the `blackbox differentiation technique <https://arxiv.org/abs/1912.02175>`_.
    """
    def __init__(self):
        super(HammingLoss, self).__init__()

    def forward(self, pred_perm: Tensor, gt_perm: Tensor) -> Tensor:
        r"""
        :param pred_perm: :math:`(b\times n_1 \times n_2)` predicted permutation matrix :math:`(\mathbf{X})`
        :param gt_perm: :math:`(b\times n_1 \times n_2)` ground truth permutation matrix :math:`(\mathbf{X}^{gt})`
        :return:
        """
        errors = pred_perm * (1.0 - gt_perm) + (1.0 - pred_perm) * gt_perm
        return errors.mean(dim=0).sum()
