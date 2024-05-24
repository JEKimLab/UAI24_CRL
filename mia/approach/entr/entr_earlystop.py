from mia.approach.entr.entr import EntrVanilla
from mia.vanilla_earlystop import VanillaMia_ES


class EntrES(EntrVanilla, VanillaMia_ES):
    def __init__(
            self,
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref=None, shadow_ref=None
    ):
        super(EntrES, self).__init__(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref, shadow_ref
        )


if __name__ == '__main__':
    pass
