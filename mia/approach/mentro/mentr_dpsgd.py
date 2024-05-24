from mia.approach.mentro.mentr import MentrVanilla
from mia.vanilla_dpsgd import VanillaMia_DP_SGD


class MentrDPSGD(MentrVanilla, VanillaMia_DP_SGD):
    def __init__(
            self,
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref=None, shadow_ref=None
    ):
        super(MentrDPSGD, self).__init__(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref, shadow_ref
        )


if __name__ == '__main__':
    pass
