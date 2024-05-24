import sys


def get_mia(
        args,
        target_model, shadow_models, attack_model,
        target_train, target_test, shadow_train_list, shadow_test_list,
        target_ref=None, shadow_ref_list=None
):
    if args.learn_method == 'vanilla':
        from mia.approach.gradw.gradw import Gradw
        func = Gradw(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list
        )
    elif args.learn_method == 'advreg':
       from mia.approach.gradw.gradw_advreg import GradwAdvReg
       func = GradwAdvReg(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list,
           target_ref, shadow_ref_list
       )
    elif args.learn_method == 'relaxloss':
       from mia.approach.gradw.gradw_relaxloss import GradwRelaxLoss
       func = GradwRelaxLoss(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list
       )
    elif args.learn_method == 'crl':
       from mia.approach.gradw.gradw_crl import GradwCRL
       func = GradwCRL(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list
       )
    elif args.learn_method == 'earlystopping':
       from mia.approach.gradw.gradw_earlystop import GradwES
       func = GradwES(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list
       )
    elif args.learn_method == 'labelsmoothing':
       from mia.approach.gradw.gradw_labelsmoothing import GradwLS
       func = GradwLS(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list
       )
    elif args.learn_method == 'confidence_penalty':
       from mia.approach.gradw.gradw_cp import GradwCP
       func = GradwCP(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list
       )
    #elif args.learn_method == 'dpsgd':
    #   from mia.approach.entr.entr_dpsgd import EntrDPSGD
    #   func = EntrDPSGD(
    #       target_model, shadow_models, attack_model,
    #       target_train, target_test, shadow_train_list, shadow_test_list
    #   )
    else:
        print('the learn method name you have entered is not supported in {} yet'.format(args.learn_method))
        NotImplementedError()
        sys.exit()
    return func
