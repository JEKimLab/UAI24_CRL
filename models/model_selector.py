import sys


def get_network(args):
    """ return given network
    """
    print('Selecting networks')
    # CIFAR-100 or CIFAR-10 or SVHN
    if args.dataset in ['cifar10', 'cifar10_test', 'svhn', 'svhn_test']:
        import models.cifar.model_selector_cifar as selector
        return selector.get_network(args, num_classes=10)
    elif args.dataset == 'cifar100' or args.dataset == 'cifar100_test':
        import models.cifar.model_selector_cifar as selector
        return selector.get_network(args, num_classes=100)
    elif args.dataset == 'arxiv10' or args.dataset == 'arxiv10_test':
        import models.arxiv10.model_selector_arxiv as selector
        return selector.get_network(args, num_classes=10)
    # Not Supported
    else:
        print_not_support(args.dataset)


def print_not_support(dataset):
    """ return unsupported info
    """
    print('the network name you have entered is not supported in {} yet'.format(dataset))
    sys.exit()
