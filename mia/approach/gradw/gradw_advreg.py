from mia.approach.gradw.gradw import Gradw
from mia.vanilla_advreg import VanillaMia_AdvReg


class GradwAdvReg(Gradw, VanillaMia_AdvReg):
    def __init__(
            self,
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref=None, shadow_ref=None
    ):
        super(GradwAdvReg, self).__init__(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref, shadow_ref
        )


if __name__ == '__main__':
    pass
