import os
import yaml

from conf import settings


def load_model_conf(arch, conf_file):
    """ For loading model conf files in ./conf/model
    """
    conf_path = os.path.join(settings.CONF_MODEL_DIR, arch, '{}.yml'.format(conf_file))
    with open(file=conf_path, mode="rb") as f:
        infos = yaml.load(f, Loader=yaml.FullLoader)
    return infos


def load_dataset_conf(dataset, conf_file):
    """ For loading data files
    """
    conf_path = os.path.join(settings.CONF_DIR, '{}.yml'.format(conf_file))
    with open(file=conf_path, mode="rb") as f:
        infos = yaml.load(f, Loader=yaml.FullLoader)
    return infos


def load_loss_conf(conf_file):
    """ For loading loss conf files in ./conf/loss
    """
    conf_path = os.path.join(settings.CONF_LOSS_DIR, '{}.yml'.format(conf_file))
    with open(file=conf_path, mode="rb") as f:
        infos = yaml.load(f, Loader=yaml.FullLoader)
    return infos
