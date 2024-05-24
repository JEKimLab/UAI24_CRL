from mia.approach.gradx.gradx import Gradx
from mia.vanilla_crl import VanillaMia_CRL


class GradxCRL(Gradx, VanillaMia_CRL):
    def __init__(
            self,
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref=None, shadow_ref=None
    ):
        super(GradxCRL, self).__init__(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref, shadow_ref
        )


if __name__ == '__main__':
    pass
