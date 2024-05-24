from mia.approach.mentro.mentr import MentrVanilla
from mia.vanilla_confidencepenalty import VanillaMia_CP


class MentrCP(MentrVanilla, VanillaMia_CP):
    def __init__(
            self,
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref=None, shadow_ref=None
    ):
        super(MentrCP, self).__init__(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref, shadow_ref
        )


if __name__ == '__main__':
    pass
