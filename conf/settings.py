import os

"""
For path settings
"""
PROJECT_DIR = os.getcwd()

# conf path
CONF_DIR = f"{PROJECT_DIR}/conf/dataset"
CONF_MODEL_DIR = f"{PROJECT_DIR}/conf/model"
CONF_LOSS_DIR = f"{PROJECT_DIR}/conf/loss"
CONF_TRAIN_DIR = f"{PROJECT_DIR}/conf/train"
CONF_RETRAIN_DIR = f"{PROJECT_DIR}/conf/retrain"
CONF_TRAIN_ATTACK_DIR = f"{PROJECT_DIR}/conf/train_attack"

# set Dataset path here
DATASET_DIR = '/home/ubuntu/dataset'

if __name__ == '__main__':
    path = os.getcwd()
    print(path)
    print(PROJECT_DIR)
    print(CONF_TRAIN_DIR)
    print(DATASET_DIR)
