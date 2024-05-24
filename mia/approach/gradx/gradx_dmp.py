from mia.approach.gradx.gradx_relaxloss import Gradx
from mia.vanilla_dmp import VanillaMia_DMP


class GradxDMP(Gradx, VanillaMia_DMP):
    def __init__(
            self,
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref=None, shadow_ref=None
    ):
        super(GradxDMP, self).__init__(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref, shadow_ref
        )


if __name__ == '__main__':
    pass
