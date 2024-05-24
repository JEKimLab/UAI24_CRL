from mia.approach.mentro.mentr import MentrVanilla
from mia.vanilla_labelsmoothing import VanillaMia_LS


class MentrLS(MentrVanilla, VanillaMia_LS):
    def __init__(
            self,
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref=None, shadow_ref=None
    ):
        super(MentrLS, self).__init__(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref, shadow_ref
        )


if __name__ == '__main__':
    pass
