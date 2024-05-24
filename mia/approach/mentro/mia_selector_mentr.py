
import sys


def get_mia(
        args,
        target_model, shadow_models, attack_model,
        target_train, target_test, shadow_train_list, shadow_test_list,
        target_ref=None, shadow_ref_list=None
):
    if args.learn_method == 'vanilla':
        from mia.approach.mentro.mentr import MentrVanilla
        func = MentrVanilla(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list
        )
    elif args.learn_method == 'advreg':
       from mia.approach.mentro.mentr_advreg import MentrAdvReg
       func = MentrAdvReg(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list,
           target_ref, shadow_ref_list
       )
    elif args.learn_method == 'dmp':
       from mia.approach.mentro.mentr_dmp import MentrDMP
       func = MentrDMP(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list,
           target_ref, shadow_ref_list
       )
    elif args.learn_method == 'dmpgan':
       from mia.approach.mentro.mentr_dmp_gan import MentrDMPGAN
       func = MentrDMPGAN(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list,
       )
    elif args.learn_method == 'relaxloss':
       from mia.approach.mentro.mentr_relaxloss import MentrRelaxLoss
       func = MentrRelaxLoss(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list
       )
    elif args.learn_method == 'crl':
       from mia.approach.mentro.mentr_crl import MentrCRL
       func = MentrCRL(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list
       )
    elif args.learn_method == 'earlystopping':
       from mia.approach.mentro.mentr_earlystop import MentrES
       func = MentrES(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list
       )
    elif args.learn_method == 'labelsmoothing':
        from mia.approach.mentro.mentr_labelsmoothing import MentrLS
        func = MentrLS(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list
        )
    elif args.learn_method == 'confidence_penalty':
        from mia.approach.mentro.mentr_confidencepenalty import MentrCP
        func = MentrCP(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list
        )
    elif args.learn_method == 'dpsgd':
       from mia.approach.mentro.mentr_dpsgd import MentrDPSGD
       func = MentrDPSGD(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list
       )
    else:
        print('the learn method name you have entered is not supported in {} yet'.format(args.learn_method))
        NotImplementedError()
        sys.exit()
    return func
