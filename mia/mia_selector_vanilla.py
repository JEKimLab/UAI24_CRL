import sys


def get_mia(
        args,
        target_model, shadow_models, attack_model,
        target_train, target_test, shadow_train_list, shadow_test_list,
        target_ref=None, shadow_ref_list=None
):
    if args.learn_method == 'vanilla':
        from mia.vanilla import VanillaMia
        func = VanillaMia(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list
        )
    elif args.learn_method == 'advreg':
        from mia.vanilla_advreg import VanillaMia_AdvReg
        func = VanillaMia_AdvReg(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref, shadow_ref_list
        )
    elif args.learn_method == 'dmp':
        from mia.vanilla_dmp import VanillaMia_DMP
        func = VanillaMia_DMP(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref, shadow_ref_list
        )
    elif args.learn_method == 'dmpgan':
        from mia.vanilla_dmp_gan import VanillaMia_DMP_GAN
        func = VanillaMia_DMP_GAN(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list
        )
    elif args.learn_method == 'relaxloss':
        from mia.vanilla_relaxloss import VanillaMia_RelaxLoss
        func = VanillaMia_RelaxLoss(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list
        )
    elif args.learn_method == 'crl':
        from mia.vanilla_crl import VanillaMia_CRL
        func = VanillaMia_CRL(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list
        )
    elif args.learn_method == 'earlystopping':
        from mia.vanilla_earlystop import VanillaMia_ES
        func = VanillaMia_ES(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list
        )
    elif args.learn_method == 'labelsmoothing':
        from mia.vanilla_labelsmoothing import VanillaMia_LS
        func = VanillaMia_LS(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list
        )
    elif args.learn_method == 'confidence_penalty':
        from mia.vanilla_confidencepenalty import VanillaMia_CP
        func = VanillaMia_CP(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list
        )
    elif args.learn_method == 'dpsgd':
        from mia.vanilla_dpsgd import VanillaMia_DP_SGD
        func = VanillaMia_DP_SGD(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list
        )
    elif args.learn_method == 'centerloss':
        from mia.vanilla_centerloss import VanillaMia_Centerloss
        func = VanillaMia_Centerloss(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list
        )
    else:
        print('the learn method name you have entered is not supported in {} yet'.format(args.learn_method))
        NotImplementedError()
        sys.exit()
    return func
