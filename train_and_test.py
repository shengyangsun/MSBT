from sklearn.metrics import auc, precision_recall_curve
import torch
import numpy as np
import math
from tqdm import tqdm

def MSBT_train(args, dataloader, model_MT, optimizer_MT, criterion, logger):
    with torch.set_grad_enabled(True):
        model_MT.train()
        for i, (f_v, f_a, f_f, label) in enumerate(dataloader):
            seq_len = torch.sum(torch.max(torch.abs(f_v), dim=2)[0] > 0, 1)
            f_v = f_v[:, :torch.max(seq_len), :]
            f_a = f_a[:, :torch.max(seq_len), :]
            f_f = f_f[:, :torch.max(seq_len), :]
            f_v, f_a, f_f, label = f_v.float().cuda(), f_a.float().cuda(), f_f.float().cuda(), label.float().cuda()
            MIL_logits, loss_TCC = model_MT(f_a, f_v, f_f, seq_len)
            loss_MIL = criterion(MIL_logits, label)
            loss_TCC = args.lambda_infoNCE * loss_TCC
            total_loss = loss_MIL + loss_TCC
            logger.info(f"Current batch: {i}, Loss: {total_loss:.4f}, MIL: {loss_MIL:.4f}, TCC: {loss_TCC:.4f}")
            optimizer_MT.zero_grad()
            total_loss.backward()
            optimizer_MT.step()


def MSBT_test(dataloader, model_MT, gt):
    with torch.no_grad():
        model_MT.eval()
        pred = torch.zeros(0).cuda()
        for i, (f_v, f_a, f_f) in tqdm(enumerate(dataloader)):
            f_v, f_a, f_f = f_v.cuda(), f_a.cuda(), f_f.cuda()
            logits, _ = model_MT(f_a, f_v, f_f, seq_len=None)
            logits = torch.mean(logits, 0)
            pred = torch.cat((pred, logits))
        pred = list(pred.cpu().detach().numpy())
        precision, recall, th = precision_recall_curve(list(gt), np.repeat(pred, 16))
        ap = auc(recall, precision)
        return ap
