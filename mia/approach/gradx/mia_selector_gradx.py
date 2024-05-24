import sys


def get_mia(
        args,
        target_model, shadow_models, attack_model,
        target_train, target_test, shadow_train_list, shadow_test_list,
        target_ref=None, shadow_ref_list=None
):
    if args.learn_method == 'vanilla':
        from mia.approach.gradx.gradx import Gradx
        func = Gradx(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list
        )
    elif args.learn_method == 'advreg':
       from mia.approach.gradx.gradx_advreg import GradxAdvReg
       func = GradxAdvReg(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list,
           target_ref, shadow_ref_list
       )
    elif args.learn_method == 'dmp':
       from mia.approach.gradx.gradx_dmp import GradxDMP
       func = GradxDMP(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list,
           target_ref, shadow_ref_list
       )
    elif args.learn_method == 'dmpgan':
       from mia.approach.gradx.gradx_dmp_gan import GradxDMPGAN
       func = GradxDMPGAN(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list,
       )
    elif args.learn_method == 'relaxloss':
       from mia.approach.gradx.gradx_relaxloss import GradxRelaxLoss
       func = GradxRelaxLoss(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list
       )
    elif args.learn_method == 'crl':
       from mia.approach.gradx.gradx_crl import GradxCRL
       func = GradxCRL(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list
       )
    elif args.learn_method == 'earlystopping':
       from mia.approach.gradx.gradx_earlystop import GradxES
       func = GradxES(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list
       )
    elif args.learn_method == 'labelsmoothing':
       from mia.approach.gradx.gradx_labelsmoothing import GradxLS
       func = GradxLS(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list
       )
    elif args.learn_method == 'confidence_penalty':
       from mia.approach.gradx.gradx_cp import GradxCP
       func = GradxCP(
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
