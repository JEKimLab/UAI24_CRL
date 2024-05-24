import torch
import torch.utils.data as data

from mia.base import BaseMia


class VanillaMia(BaseMia):
    def __init__(
            self,
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref=None, shadow_ref=None
    ):
        super(VanillaMia, self).__init__(target_model, shadow_models, attack_model)
        self.load_dataset(target_train, target_test, shadow_train_list, shadow_test_list)

    def train_attack_model(self, args):
        if self.attack_train is None:
            self.produce_attack(args)
        super().train_attack_model(args)

    @torch.no_grad()
    def produce_attack(self, args):
        """ make attack dataset """
        target_model = self.target_model.cpu()
        for i in range(len(self.shadow_models)):
            self.shadow_models[i] = self.shadow_models[i].cpu()
        ''' attack training data '''
        data_shadow_train_list = []
        data_shadow_test_list = []
        data_shadow_train_class_list = []
        data_shadow_test_class_list = []
        for i in range(len(self.shadow_models)):
            cur_shadow_model = self.shadow_models[i]
            cur_train_loader = self.shadow_train_list[i]
            cur_test_loader = self.shadow_test_list[i]
            preds, targets = self.get_predictions(args, cur_shadow_model, cur_train_loader)
            data_shadow_train_list += preds
            data_shadow_train_class_list += targets
            preds, targets = self.get_predictions(args, cur_shadow_model, cur_test_loader)
            data_shadow_test_list += preds
            data_shadow_test_class_list += targets

        # print(len(data_shadow_train_list), len(data_shadow_test_list), len(data_shadow_train_class_list), len(data_shadow_test_class_list))
        ''' attack testing data '''
        data_target_train_list = []
        data_target_test_list = []
        data_target_train_class_list = []
        data_target_test_class_list = []
        preds, targets = self.get_predictions(args, target_model, self.target_train)
        data_target_train_list += preds
        data_target_train_class_list += targets
        preds, targets = self.get_predictions(args, target_model, self.target_test)
        data_target_test_list += preds
        data_target_test_class_list += targets
        ''' load dataloader '''
        from dataloader.info import num_classes
        from dataloader.attack.data_attack_vanilla import MIADataset
        attack_train_dataset = MIADataset(
            data_shadow_train_list, data_shadow_test_list,
            data_shadow_train_class_list, data_shadow_test_class_list,
            num_classes=num_classes[args.dataset]
        )
        attack_test_dataset = MIADataset(
            data_target_train_list, data_target_test_list,
            data_target_train_class_list, data_target_test_class_list,
            num_classes=num_classes[args.dataset]
        )
        self.attack_train = data.DataLoader(
            attack_train_dataset,
            batch_size=args.attack_batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=False
        )
        self.attack_test = data.DataLoader(
            attack_test_dataset,
            batch_size=args.attack_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False
        )

    @torch.no_grad()
    def produce_attack_test(self, args):
        """ make attack dataset """
        target_model = self.target_model.cpu()
        ''' attack testing data '''
        data_target_train_list = []
        data_target_test_list = []
        data_target_train_group_list = []
        data_target_test_group_list = []
        data_target_train_class_list = []
        data_target_test_class_list = []
        preds, targets = self.get_predictions_test(args, target_model, self.target_train)
        #preds, targets, groups = self.get_predictions_test(args, target_model, self.target_train)
        data_target_train_list += preds
        #data_target_train_group_list += groups
        data_target_train_class_list += targets
        preds, targets = self.get_predictions_test(args, target_model, self.target_test)
        #preds, targets, groups = self.get_predictions_test(args, target_model, self.target_test)
        data_target_test_list += preds
        #data_target_test_group_list += groups
        data_target_test_class_list += targets
        ''' load dataloader '''
        from dataloader.info import num_classes
        from dataloader.attack.data_attack_vanilla import MIADataset
        attack_test_dataset = MIADataset(
            data_target_train_list, data_target_test_list,
            data_target_train_class_list, data_target_test_class_list,
            num_classes=num_classes[args.dataset],
            #if_group=True,
            #data_train_group_list=None,#data_target_train_group_list,
            #data_test_group_list=None#data_target_test_group_list
        )
        self.attack_test = data.DataLoader(
            attack_test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False
        )


if __name__ == '__main__':
    a = VanillaMia(None, None, None, None, None, None, None)
