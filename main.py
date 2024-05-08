from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import time
import math
import numpy as np
import random
import os
from MultimodalTransformer import MultimodalTransformer
from train_and_test import MSBT_train as train
from train_and_test import MSBT_test as test
import option
from utils import Prepare_logger
from load_dataset import Dataset

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    global logger
    torch.multiprocessing.set_start_method('spawn')
    args = option.parser.parse_args()
    setup_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    train_loader = DataLoader(Dataset(args, test_mode=False),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=5, shuffle=False,
                             num_workers=args.workers, pin_memory=True)
    model_MT = MultimodalTransformer(args).cuda()
    gt = np.load(args.gt)

    if args.eval:
        state_dict = torch.load(args.model_path)
        model_MT.load_state_dict(state_dict, True)
        model_MT.eval()
        test_ap = test(test_loader, model_MT, gt)
        print ('Test AP: {:.4}'.format(test_ap))
    else:
        if not os.path.exists('./log'):
            os.makedirs('./log')
        if not os.path.exists('./ckpt'):
            os.makedirs('./ckpt')

        logger = Prepare_logger()
        logger.info(args)

        criterion = torch.nn.BCELoss()
        optimizer_MT = optim.SGD(model_MT.parameters(), lr=args.lr, weight_decay=0.0005)

        best_test_ap = 0
        best_epoch = 0
        for epoch in range(args.max_epoch):
            st = time.time()
            train(args, train_loader, model_MT, optimizer_MT, criterion, logger)
            test_ap = test(test_loader, model_MT, gt)
            if test_ap > best_test_ap:
                best_test_ap = test_ap
                best_epoch = epoch
                torch.save(model_MT.state_dict(), './ckpt/' + args.model_name + '_best.pkl')
            logger.info('Epoch {}/{}: AP:{:.4}\n'.format(epoch, args.max_epoch, test_ap))
            logger.info('Best Performance in Epoch {}: AP:{:.4}\n'.format(best_epoch, best_test_ap))
