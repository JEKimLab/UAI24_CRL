"""
these functions are mainly derived from: https://github.com/ganeshdg95/Leveraging-Adversarial-Examples-to-Quantify-Membership-Information-Leakage/blob/main/utils.py
"""
import torch

from autoattack import AutoAttack

cuda = True if torch.cuda.is_available() else False


def advDistance(model, images, labels, batch_size=10, epsilon=1, norm='Linf'):
    """ Computes the adversarial distance score. First, adversarial examples are computed for each sample that is
    correctly classified by the target model. Then, the distance between the original and adversarial samples is
    computed. If a sample is misclassified, resulting adversarial distance will be 0.
    model: instance of a nn.Module subclass
    images: pytorch tensor with dimensions [batch,channels,width,height]
    labels: pytorch tensor of shape [batch] containing the integer labels of the 'images'
    batch_size: integer indicating the batch size for computing adversarial examples
    epsilon: maximum value for the magnitude of perturbations
    norm: indicates the norm used for computing adversarial examples and for measuring the distance between samples.
    Must be in {'Linf','L2','L1'}
    outputs -> pytorch tensor of dimensions [batch] containing the adversarial distance of 'images'
    """
    if norm == 'Linf':
        ordr = float('inf')
    elif norm == 'L1':
        ordr = 1
    elif norm == 'L2':
        ordr = 2
    else:
        raise NotImplementedError

    if cuda:
        dev = 'cuda'
    else:
        dev = 'cpu'

    adversary = AutoAttack(model, norm=norm, eps=epsilon, version='standard', device=dev)
    adv = adversary.run_standard_evaluation(images, labels, bs=batch_size)

    dist = Dist(images, adv, ordr=ordr)

    return dist


def Dist(sample, adv, ordr=float('inf')):
    """Computes the norm of the difference between two vectors. The operation is done for batches of vectors
    sample: pytorch tensor with dimensions [batch, others]
    adv: pytorch tensor with the same dimensions as 'sample'
    ordr: order of the norm. Must be in {1,2,float('inf')}
    outputs -> pytorch tensor of dimensions [batch] containing distance values for the batch of samples.
    """
    sus = sample - adv
    sus = sus.view(sus.shape[0], -1)
    return torch.norm(sus, ordr, 1)
