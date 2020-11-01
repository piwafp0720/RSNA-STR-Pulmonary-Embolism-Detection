import torch

from modules.base_module import BaseModule


class RSNAModule(BaseModule):
    def __init__(self, *args, **kwargs):
        self.network_type = kwargs.pop('network_type')
        assert self.network_type in ['cnn', 'rnn']

        if self.network_type == 'rnn':
            self.n_classes = kwargs.pop('n_classes')

        super().__init__(*args, **kwargs)


    def training_step(self, batch, batch_nb):
        if self.network_type == 'cnn':
            return self._training_step_image_level(batch, batch_nb)
        elif self.network_type == 'rnn':
            return self._training_step_exam_level(batch, batch_nb)

    def _training_step_image_level(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)

        loss = self.loss(y, t)

        pred = y.sigmoid().detach()
        target = t.round()
        acc = (pred.round() == target).float().mean()

        output = {
            'loss': loss,
            'progress_bar': {
                'acc': acc
            },
            'log': {
                'train/loss': loss,
                'train/acc': acc,
            }
        }

        return output

    def _training_step_exam_level(self, batch, batch_nb):
        x, t, sequence_length = batch
        y_image_level, y_exam_level = self.forward(x, sequence_length)

        t_image_level = t[:, :, -1] # (batchsize, max_sequence)
        t_exam_level = t[:, 0, :-1].squeeze(1) #(batch_size, n_class - 1)

        loss_image_level = self.loss(
            y_image_level[t_image_level > -1], t_image_level[t_image_level > -1]).mean()
        
        loss_exam_level_each = self.loss(y_exam_level, t_exam_level)
        
        loss_exam_level = loss_exam_level_each.mean()

        loss = (loss_exam_level + loss_image_level) / 2

        pred_image_level = y_image_level.sigmoid().detach()
        target_image_level = t_image_level.round()
        acc_image_level = []
        for p, gt, seq in zip(pred_image_level, target_image_level, sequence_length):
            acc = (p[:seq].round() == gt[:seq]).float()
            acc_image_level.append(acc)
        acc_image_level = torch.cat(acc_image_level).mean()

        pred_exam_level = y_exam_level.sigmoid().detach()
        target_exam_level = t_exam_level.round()
        acc_exam_level = (pred_exam_level.round() 
            == target_exam_level).float().mean()
        
        output = {
            'loss': loss,
            'progress_bar': {
                'acc_image_level': acc_image_level,
                'acc_exam_level': acc_exam_level,
            },
            'log':{
                'train/loss': loss,
                'train/acc_image_level': acc_image_level,
                'train/acc_exam_level': acc_exam_level,
                'train/loss_negative_exam_for_pe': loss_exam_level_each[:, 0].mean(),
                'train/loss_indeterminate': loss_exam_level_each[:, 1].mean(),
                'train/loss_chronic_pe': loss_exam_level_each[:, 2].mean(),
                'train/loss_acute_and_chronic_pe': loss_exam_level_each[:, 3].mean(),
                'train/loss_central_pe': loss_exam_level_each[:, 4].mean(),
                'train/loss_leftside_pe': loss_exam_level_each[:, 5].mean(),
                'train/loss_rightside_pe': loss_exam_level_each[:, 6].mean(),
                'train/loss_rv_lv_ratio_gte_1': loss_exam_level_each[:, 7].mean(),
                'train/loss_rv_lv_ratio_lt_1': loss_exam_level_each[:, 8].mean(),
                'train/loss_pe_present_on_image': loss_image_level,
            }
        }

        return output

    def validation_step(self, batch, batch_nb):
        if self.network_type == 'cnn':
            return self._validation_step_image_level(batch, batch_nb)
        elif self.network_type == 'rnn':
            return self._validation_step_exam_level(batch, batch_nb)

    def _validation_step_image_level(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)

        val_batch_loss = self.loss(y, t)

        pred = y.sigmoid().detach()
        target = t.round()
        val_batch_acc = (pred.round() == target).float().mean()

        output = {
            'val_batch_loss': val_batch_loss,
            'val_batch_acc': val_batch_acc,
        }

        return output
    
    def _validation_step_exam_level(self, batch, batch_nb):
        x, t, sequence_length = batch
        y_image_level, y_exam_level = self.forward(x, sequence_length)

        t_image_level = t[:, :, -1] # (batchsize, max_sequence)
        t_exam_level = t[:, 0, :-1].squeeze(1) #(batch_size, n_class - 1)

        loss_image_level = self.loss(
            y_image_level[t_image_level > -1], t_image_level[t_image_level > -1]).mean()
        
        loss_exam_level_each = self.loss(y_exam_level, t_exam_level)
        
        loss_exam_level = loss_exam_level_each.mean()

        val_batch_loss = (loss_exam_level + loss_image_level) / 2

        pred_image_level = y_image_level.sigmoid().detach()
        target_image_level = t_image_level.round()
        val_batch_acc_image_level = []
        for p, gt, seq in zip(pred_image_level, target_image_level, sequence_length):
            acc = (p[:seq].round() == gt[:seq]).float()
            val_batch_acc_image_level.append(acc)
        val_batch_acc_image_level = torch.cat(val_batch_acc_image_level).mean()

        pred_exam_level = y_exam_level.sigmoid().detach()
        target_exam_level = t_exam_level.round()
        val_batch_acc_exam_level = (pred_exam_level.round() 
            == target_exam_level).float().mean()

        output = {
            'val_batch_loss': val_batch_loss,
            'val_batch_acc_image_level': val_batch_acc_image_level,
            'val_batch_acc_exam_level': val_batch_acc_exam_level,
            'val_batch_negative_exam_for_pe': loss_exam_level_each[:, 0].mean(),
            'val_batch_indeterminate': loss_exam_level_each[:, 1].mean(),
            'val_batch_chronic_pe': loss_exam_level_each[:, 2].mean(),
            'val_batch_acute_and_chronic_pe': loss_exam_level_each[:, 3].mean(),
            'val_batch_central_pe': loss_exam_level_each[:, 4].mean(),
            'val_batch_leftside_pe': loss_exam_level_each[:, 5].mean(),
            'val_batch_rightside_pe': loss_exam_level_each[:, 6].mean(),
            'val_batch_rv_lv_ratio_gte_1': loss_exam_level_each[:, 7].mean(),
            'val_batch_rv_lv_ratio_lt_1': loss_exam_level_each[:, 8].mean(),
            'val_batch_pe_present_on_image': loss_image_level,
        }
        return output

    def validation_epoch_end(self, outputs):
        if self.network_type == 'cnn':
            return self._validation_epoch_end_image_level(outputs)
        elif self.network_type == 'rnn':
            return self._validation_epoch_end_exam_level(outputs)
    
    def _validation_epoch_end_image_level(self, outputs):
        val_loss = torch.stack(
            [output['val_batch_loss'] for output in outputs]).mean()
        val_acc = torch.stack(
            [output['val_batch_acc'] for output in outputs]).mean()
        
        results = {
            'val_loss': val_loss,
            'progress_bar': {
                'val_loss': val_loss,
                'val_acc': val_acc,
            },
            'log': {
                'val/loss': val_loss,
                'val/acc': val_acc,
            }
        }

        return results
    
    def _validation_epoch_end_exam_level(self, outputs):
        val_loss = torch.stack(
            [output['val_batch_loss'] for output in outputs]).mean()
        val_acc_image_level = torch.stack(
            [output['val_batch_acc_image_level'] for output in outputs]).mean()
        val_acc_exam_level = torch.stack(
            [output['val_batch_acc_exam_level'] for output in outputs]).mean()
        val_loss_negative_exam_for_pe = torch.stack(
            [output['val_batch_negative_exam_for_pe'] for output in outputs]).mean()
        val_loss_indeterminate = torch.stack(
            [output['val_batch_indeterminate'] for output in outputs]).mean()
        val_loss_chronic_pe = torch.stack(
            [output['val_batch_chronic_pe'] for output in outputs]).mean()
        val_loss_acute_and_chronic_pe = torch.stack(
            [output['val_batch_acute_and_chronic_pe'] for output in outputs]).mean()
        val_loss_central_pe = torch.stack(
            [output['val_batch_central_pe'] for output in outputs]).mean()
        val_loss_leftside_pe = torch.stack(
            [output['val_batch_leftside_pe'] for output in outputs]).mean()
        val_loss_rightside_pe = torch.stack(
            [output['val_batch_rightside_pe'] for output in outputs]).mean()
        val_loss_rv_lv_ratio_gte_1 = torch.stack(
            [output['val_batch_rv_lv_ratio_gte_1'] for output in outputs]).mean()
        val_loss_rv_lv_ratio_lt_1 = torch.stack(
            [output['val_batch_rv_lv_ratio_lt_1'] for output in outputs]).mean()
        val_loss_pe_present_on_image = torch.stack(
            [output['val_batch_pe_present_on_image'] for output in outputs]).mean()
        
        results = {
            'val_loss': val_loss,
            'progress_bar': {
                'val_loss': val_loss,
                'val_acc_image_level': val_acc_image_level,
                'val_acc_exam_level': val_acc_exam_level,
            },
            'log': {
                'val/loss': val_loss,
                'val/acc_image_level': val_acc_image_level,
                'val/acc_exam_level': val_acc_exam_level,
                'val/loss_negative_exam_for_pe': val_loss_negative_exam_for_pe,
                'val/loss_indeterminate': val_loss_indeterminate,
                'val/loss_chronic_pe': val_loss_chronic_pe,
                'val/loss_acute_and_chronic_pe': val_loss_acute_and_chronic_pe,
                'val/loss_central_pe': val_loss_central_pe,
                'val/loss_leftside_pe': val_loss_leftside_pe,
                'val/loss_rightside_pe': val_loss_rightside_pe,
                'val/loss_rv_lv_ratio_gte_1': val_loss_rv_lv_ratio_gte_1,
                'val/loss_rv_lv_ratio_lt_1': val_loss_rv_lv_ratio_lt_1,
                'val/loss_pe_present_on_image': val_loss_pe_present_on_image,
            }
        }

        return results
    
