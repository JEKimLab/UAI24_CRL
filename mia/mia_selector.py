import sys


def get_mia(
        args,
        target_model, shadow_models, attack_model,
        target_train, target_test, shadow_train_list, shadow_test_list,
        target_ref=None, shadow_ref_list=None
):
    if args.attack_method == 'vanilla':
        from mia.mia_selector_vanilla import get_mia as gm
        func = gm(
            args, target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref, shadow_ref_list
        )
    elif args.attack_method == 'entr':
        from mia.approach.entr.mia_selector_entr import get_mia as gm
        func = gm(
            args, target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref, shadow_ref_list
        )
    elif args.attack_method == 'mentr':
        from mia.approach.mentro.mia_selector_mentr import get_mia as gm
        func = gm(
            args, target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref, shadow_ref_list
        )
    elif args.attack_method == 'gradx':
        from mia.approach.gradx.mia_selector_gradx import get_mia as gm
        func = gm(
            args, target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref, shadow_ref_list
        )
    elif args.attack_method == 'gradw':
        from mia.approach.gradw.mia_selector_gradw import get_mia as gm
        func = gm(
            args, target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref, shadow_ref_list
        )
    elif args.attack_method == 'advdist':
        print('the attack method name you have entered is not supported in {} yet'.format(args.attack_method))
        NotImplementedError()
        sys.exit()
    else:
        print('the attack method name you have entered is not supported in {} yet'.format(args.attack_method))
        NotImplementedError()
        sys.exit()
    return func
