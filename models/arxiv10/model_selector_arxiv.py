import sys

from util.yaml_loader import load_model_conf
from conf.settings import DATASET_DIR

def get_network(args, num_classes=10):
    """ return given network
    """
    print(args.arch, args.arch_conf)
    info = load_model_conf(args.arch, args.arch_conf)
    print(info)
    # HAN
    if args.arch == 'han':
        from models.arxiv10.han.hierarchical_att_model import han
        net = han(
            info['words_dim'], info['word_num_hidden'],
            info['sentence_num_hidden'], f"{DATASET_DIR}/ArXiv-10/{info['pretrained_word2vec_path']}",
            num_classes=num_classes
        )
    # Not Supported
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    return net
