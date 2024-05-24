import sys


def get_mia(
        args,
        target_model, shadow_models, attack_model,
        target_train, target_test, shadow_train_list, shadow_test_list,
        target_ref=None, shadow_ref_list=None
):
    if args.learn_method == 'vanilla':
        from mia.approach.entr.entr import EntrVanilla
        func = EntrVanilla(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list
        )
    elif args.learn_method == 'advreg':
       from mia.approach.entr.entr_advreg import EntrAdvReg
       func = EntrAdvReg(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list,
           target_ref, shadow_ref_list
       )
    elif args.learn_method == 'dmp':
       from mia.approach.entr.entr_dmp import EntrDMP
       func = EntrDMP(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list,
           target_ref, shadow_ref_list
       )
    elif args.learn_method == 'dmpgan':
       from mia.approach.entr.entr_dmp_gan import EntrDMPGAN
       func = EntrDMPGAN(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list
       )
    elif args.learn_method == 'relaxloss':
       from mia.approach.entr.entr_relaxloss import EntrRelaxLoss
       func = EntrRelaxLoss(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list
       )
    elif args.learn_method == 'crl':
       from mia.approach.entr.entr_crl import EntrCRL
       func = EntrCRL(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list
       )
    elif args.learn_method == 'earlystopping':
       from mia.approach.entr.entr_earlystop import EntrES
       func = EntrES(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list
       )
    elif args.learn_method == 'labelsmoothing':
       from mia.approach.entr.entr_labelsmoothing import EntrLS
       func = EntrLS(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list
       )
    elif args.learn_method == 'confidence_penalty':
       from mia.approach.entr.entr_confidencepenalty import EntrCP
       func = EntrCP(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list
       )
    elif args.learn_method == 'dpsgd':
       from mia.approach.entr.entr_dpsgd import EntrDPSGD
       func = EntrDPSGD(
           target_model, shadow_models, attack_model,
           target_train, target_test, shadow_train_list, shadow_test_list
       )
    else:
        print('the learn method name you have entered is not supported in {} yet'.format(args.learn_method))
        NotImplementedError()
        sys.exit()
    return func
