import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from model import pad_layer
import numpy as np
import os
from classify_dataset import ClassifyDataset
from utils import * 
import argparse
#from preprocess.tacotron import utils

class SpeakerClassifier(nn.Module):
    def __init__(self, n_class, c_in=512, c_h=512, dp=0.4, ns=0.01):
        super(SpeakerClassifier, self).__init__()
        self.dp, self.ns = dp, ns
        self.conv1 = nn.Conv1d(c_in, c_h, kernel_size=5)
        self.conv2 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv3 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv4 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv5 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv6 = nn.Conv1d(c_h, c_h, kernel_size=5)
        self.conv7 = nn.Conv1d(c_h, c_h//2, kernel_size=3)
        self.conv8 = nn.Conv1d(c_h//2, c_h//4, kernel_size=3)
        self.conv9 = nn.Conv1d(c_h//4, n_class, kernel_size=16)
        self.drop1 = nn.Dropout(p=dp)
        self.drop2 = nn.Dropout(p=dp)
        self.drop3 = nn.Dropout(p=dp)
        self.drop4 = nn.Dropout(p=dp)
        self.ins_norm1 = nn.InstanceNorm1d(c_h)
        self.ins_norm2 = nn.InstanceNorm1d(c_h)
        self.ins_norm3 = nn.InstanceNorm1d(c_h)
        self.ins_norm4 = nn.InstanceNorm1d(c_h//4)

    def conv_block(self, x, conv_layers, after_layers, res=True):
        out = x
        for layer in conv_layers:
            out = pad_layer(out, layer)
            out = F.leaky_relu(out, negative_slope=self.ns)
        for layer in after_layers:
            out = layer(out)
        if res:
            out = out + x
        return out

    def forward(self, x):
        out = self.conv_block(x, [self.conv1, self.conv2], [self.ins_norm1, self.drop1], res=False)
        out = self.conv_block(out, [self.conv3, self.conv4], [self.ins_norm2, self.drop2], res=True)
        out = self.conv_block(out, [self.conv5, self.conv6], [self.ins_norm3, self.drop3], res=True)
        out = self.conv_block(out, [self.conv7, self.conv8], [self.ins_norm4, self.drop4], res=False)
        out = self.conv9(out)
        out = out.view(out.size()[0], -1)
        return out


class Classifier(object):
    def __init__(self, data_dir, log_dir='./log/'):
        self.get_data_loaders(data_dir)
        self.n_speakers = len(self.train_dataset.speaker2id)
        self.build_model()
        self.logger = Logger(log_dir)


    def build_model(self):
        self.SpeakerClassifier = SpeakerClassifier(n_class=self.n_speakers)
        if torch.cuda.is_available():
            self.SpeakerClassifier.cuda()
        betas = (0.5, 0.9)
        self.opt = optim.Adam(self.SpeakerClassifier.parameters(), lr=1e-3, betas=betas)

    def save_model(self, model_path, iteration, acc):
        model_save_path = os.path.join(model_path, 'model-{}-{:.2f}.ckpt').format(iteration, acc*100)
        model_params = {
            'model':self.SpeakerClassifier.state_dict(),
            'opt':self.opt.state_dict()
        }
        torch.save(model_params, model_save_path)

    def load_model(self, model_path):
        print('load model from {}'.format(model_path))
        model = torch.load(model_path)
        self.SpeakerClassifier.load_state_dict(model['model'])
        self.opt.load_state_dict(model['opt'])

    def get_data_loaders(self, data_dir):
        
        self.train_dataset = ClassifyDataset(
                os.path.join(data_dir, 'dataset.hdf5'), 
                os.path.join(data_dir, 'train_data.json'), 
                segment_size=128)

        self.val_dataset = ClassifyDataset(
                os.path.join(data_dir, 'dataset.hdf5'), 
                os.path.join(data_dir, 'in_test_data.json'), 
                segment_size=128,
                speaker2id=self.train_dataset.speaker2id)

        self.train_loader = DataLoader(
                dataset=self.train_dataset, 
                batch_size=128, 
                shuffle=True, 
                num_workers=1,
                pin_memory=True)

        self.val_loader = DataLoader(
                dataset=self.val_dataset, 
                batch_size=128, 
                shuffle=False, 
                num_workers=1,
                pin_memory=True)

        self.train_iter = infinite_iter(self.train_loader)

        return

    def set_eval(self):
        self.SpeakerClassifier.eval()

    def set_train(self):
        self.SpeakerClassifier.train()

    def forward_step(self, enc):
        logits = self.SpeakerClassifier(enc)
        return logits

    def cal_loss(self, logits, y_true):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y_true)
        return loss

    def cal_acc(self, logits, y_true):
        _, ind = torch.max(logits, dim=1)
        acc = torch.sum((ind == y_true).type(torch.FloatTensor)) / y_true.size(0)
        return acc

    def valid(self):
        # input: valid data, output: (loss, acc)
        total_loss, total_acc = 0., 0.
        self.set_eval()
        for i, data in enuerate(self.val_loader):
            x = cc(data['mel'].permute(0, 2, 1))
            y = cc(data['label'])
            logits = self.SpeakerClassifier(x)
            loss = self.cal_loss(logits, y)
            acc = cal_acc(logits, y)
            total_loss += loss.item()
            total_acc += acc  
        self.set_train()
        return total_loss / len(self.val_loader), total_acc / len(self.val_loader)

    def train(self, iters, model_path, flag='train', max_grad_norm=5):
        # load hyperparams
        best_val_acc = 0.0
        for iteration in range(iters):
            data = next(self.train_iter)
            x = cc(data['mel'].permute(0, 2, 1))
            y = cc(data['label'])
            logits = self.forward_step(x)
            loss = self.cal_loss(logits, y)
            # optimize
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.SpeakerClassifier.parameters(), max_grad_norm)
            self.opt.step()
            # calculate acc
            acc = self.cal_acc(logits, y)
            # print info
            info = {
                f'{flag}/loss': loss.item(), 
                f'{flag}/acc': acc,
            }
            slot_value = (iteration + 1, iters) + tuple([value for value in info.values()])
            log = 'Iter:[%06d/%06d], loss=%.3f, acc=%.3f'
            print(log % slot_value, end='\r')
            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, iteration)

            if iteration % 1000 == 0 and iteration != 0:
                valid_loss, valid_acc = self.valid()
                # print info
                info = {
                    f'{flag}/valid_loss': valid_loss, 
                    f'{flag}/valid_acc': valid_acc,
                }
                slot_value = (iteration + 1, iters) +  tuple([value for value in info.values()])
                log = 'Iter:[%06d/%06d], valid_loss=%.3f, valid_acc=%.3f'
                print(log % slot_value)
                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, iteration)
                if valid_acc > best_val_acc: 
                    best_val_acc = valid_acc
                    self.save_model(model_path, iteration, best_val_acc)
                    print('Save Model...{:.2f} acc'.format(best_val_acc*100))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True, action='store_true')
    parser.add_argument('--load_model', default=False, action='store_true')

    parser.add_argument('-data_dir', '-d', default='/home/mlpjb04/Voice_Conversion/h5py-LibriTTS/AdaVAE_d')
    parser.add_argument('-load_model_path', default='model/')
    parser.add_argument('-output_model_path', default='model_Classifier/')
    parser.add_argument('-flag', default='classify')
    parser.add_argument('-iters', default=200000, type=int)
    args = parser.parse_args()


    classifier = Classifier(data_dir=args.data_dir)
    if args.train:
        classifier.train(args.iters, args.output_model_path, args.flag)
