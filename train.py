from metric import IOUandSek
from datasets import ChangeDetection

import numpy as np
import os
from PIL import Image
# import shutil
import torch
# import torchcontrib
from torch.nn import CrossEntropyLoss, BCELoss, DataParallel
# import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from pspnet import BaseNet
from params import *

class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainset = ChangeDetection(root= data_root, mode="train", use_pseudo_label= use_pseudo_label)
        valset = ChangeDetection(root=data_root, mode="val")
        self.trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                      pin_memory=False, num_workers=4, drop_last=True)
        self.valloader = DataLoader(valset, batch_size=val_batch_size, shuffle=False,
                                    pin_memory=True, num_workers=4, drop_last=False)

        self.model = BaseNet(len(trainset.CLASSES), lightweight)
        # if args.pretrain_from:
        #     self.model.load_state_dict(torch.load(args.pretrain_from), strict=False)

        # if args.load_from:
        #     self.model.load_state_dict(torch.load(args.load_from), strict=True)

        # if args.use_pseudo_label:
        #     weight = torch.FloatTensor([1, 1, 1, 1, 1, 1]).to(self.device)
        #     # weight = torch.FloatTensor([1, 1, 1, 1, 1, 1])
        # else:
        #     weight = torch.FloatTensor([2, 1, 2, 2, 1, 1]).to(self.device)
        #     # weight = torch.FloatTensor([2, 1, 2, 2, 1, 1])
        # self.criterion = CrossEntropyLoss(ignore_index=-1, weight=weight)
        self.criterion = CrossEntropyLoss(weight=None, reduction='mean', ignore_index=-1)
        self.criterion_bin = BCELoss(reduction='none')

        self.optimizer = Adam([{"params": [param for name, param in self.model.named_parameters()
                                           if "backbone" in name], "lr": lr},
                               {"params": [param for name, param in self.model.named_parameters()
                                           if "backbone" not in name], "lr": lr * 10.0}],
                              lr=lr, weight_decay=weight_decay)

        self.model = DataParallel(self.model).to(self.device)
        # self.model = DataParallel(self.model)

        self.iters = 0
        self.total_iters = len(self.trainloader) * epochs
        self.previous_best = 0.0
        

    def training(self):
        tbar = tqdm(self.trainloader)
        self.model.train()
        total_loss = 0.0
        total_loss_sem = 0.0
        total_loss_bin = 0.0

        for i, (img1, img2, mask1, mask2, mask_bin) in enumerate(tbar):
        # for i, (img1, img2, mask2, mask_bin) in enumerate(tbar):
            img1, img2 = img1.to(self.device), img2.to(self.device)
            mask1, mask2 = mask1.to(self.device), mask2.to(self.device)
            # mask2 = ask2
            mask_bin = mask_bin.to(self.device)

            # out1, out2, out_bin = self.model(img1, img2)
            # out2, out_bin = self.model(img1, img2)
            out1, out2, out_bin = self.model(img1, img2)
            loss1 = self.criterion(out1, mask1)
            loss2 = self.criterion(out2, mask2)
            loss_bin = self.criterion(out_bin, mask_bin)
            loss_bin[mask_bin == 0] *= 2
            # loss_bin = loss_bin.mean()

            loss = loss_bin * 2 + loss1 + loss2
            # loss = loss_bin * 2 + loss2

            total_loss_sem += loss1.item() + loss2.item()
            # total_loss_sem += loss2.item()
            total_loss_bin += loss_bin.item()
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iters += 1
            lr = 0.01
            lr = lr * (1 - self.iters / self.total_iters) ** 0.9
            self.optimizer.param_groups[0]["lr"] = lr
            self.optimizer.param_groups[1]["lr"] = lr * 10.0

            tbar.set_description("Loss: %.3f, Semantic Loss: %.3f, Binary Loss: %.3f" %
                                 (total_loss / (i + 1), total_loss_sem / (i + 1), total_loss_bin / (i + 1)))

    def validation(self):
        tbar = tqdm(self.valloader)
        self.model.eval()
        metric = IOUandSek(num_classes=len(ChangeDetection.CLASSES))

        with torch.no_grad():
            for img1, img2, mask1, mask2,mask_bin, id in tbar:
            # for img1, img2,mask2, id in tbar:
            # for img1, img2, mask2, id in tbar:
                img1, img2 = img1.to(self.device), img2.to(self.device)

                out1, out2, out_bin = self.model(img1, img2, tta)
                out1 = torch.argmax(out1, dim=1).cpu().numpy() + 1
                out2 = torch.argmax(out2, dim=1).cpu().numpy() + 1
                out_bin = (out_bin > 0.5).cpu().numpy().astype(np.uint8)
                out1[out_bin == 0] = 0
                out2[out_bin == 0] = 0

                mask1[mask_bin == 0] = 0
                mask2[mask_bin == 0] = 0

                metric.add_batch(out1, mask1.numpy())
                metric.add_batch(out2, mask2.numpy())
                # metric.add_batch(out_bin, mask_bin.numpy())

                score, miou, sek = metric.evaluate()

                tbar.set_description("Score: %.2f, IOU: %.2f, SeK: %.2f" % (score * 100.0, miou * 100.0, sek * 100.0))

        score *= 100.0
        if score >= self.previous_best:
            if self.previous_best != 0:
                model_path = "outdir/models/%s_%s_%.2f.pth" % \
                             (model, backbone, self.previous_best)
                if os.path.exists(model_path):
                    os.remove(model_path)

            torch.save(self.model.module.state_dict(), "outdir/models/%s_%s_%.2f.pth" %
                       (model, backbone, score))
            self.previous_best = score


if __name__ == "__main__":
    trainer = Trainer()

    for epoch in range(epochs):
        print("\n==> Epoches %i, learning rate = %.5f\t\t\t\t previous best = %.2f" %
              (epoch, trainer.optimizer.param_groups[0]["lr"], trainer.previous_best))
        trainer.training()
        trainer.validation()
