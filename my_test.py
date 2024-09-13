import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse

from lib.Split_label.CRNet import CRNet

from metrics.cal_score_test import metrics
from utils.dataloader import get_loader, test_dataset
from utils.eva_funcs import eval_Smeasure, eval_mae, numpy2tensor
import scipy.io as scio
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(opt):
    weights_folder = "./save_weights/CRNet/"

    # for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    # for _data_name in ['CVC-300', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    for _data_name in ['Kvasir']:
    # for _data_name in ['CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
        print('-----------starting test {}-------------'.format(_data_name))
        data_path = './data/TestDataset/{}/'.format(_data_name)
        # data_path = './data/Fold_TestDataset/Fold1_Kvasir/'
        save_path = './test_result_picture/{}/'.format(_data_name)
        os.makedirs(save_path, exist_ok=True)

        model = SL_9_baseline_EAG_EGM_CFF(in_channels=3, num_classes=1)

        model.to(device)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)

        for epoch in range(opt.epochs):  # 循环每一轮
            print('测试{}数据集的{}轮========================='.format(_data_name, epoch+95))
            weights_path = os.path.join(weights_folder, f"{opt.resultsname}{epoch+95}")
            print("权重名为：",weights_path)
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            model.to(device)  # Move the model back to the appropriate device (CPU or CUDA)
            model.eval()

            with torch.no_grad():
                test_loader = test_dataset(image_root, gt_root, opt.testsize)
                for i in range(test_loader.size):
                    # print(['--------------processing-------------', i])
                    image, gt, name = test_loader.load_data()
                    gt = np.asarray(gt, np.float32)
                    gt /= (gt.max() + 1e-8)
                    image = image.to(device)  # Move the image tensor to the appropriate device (CPU or CUDA)
                    res,eg = model(image)
                    # res = model(image)
                    # _,_,_,res = model(image)

                    # ########区域分割########
                    # res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    # ########区域分割########

                    # res = F.upsample(eg, size=gt.shape, mode='bilinear', align_corners=False)
                    # res = (((res.sigmoid())>0.01).float()).data.cpu().numpy().squeeze()
                    # res = (res - res.min()) / (res.max() - res.min() + 1e-8)

                    cv2.imwrite(save_path + name, res*255)

                print('形成图片完成！======================================')
                metrics(opt, epoch, _data_name)
                print('\n\n')
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")
    parser.add_argument("--data-path", default="./", help="DRIVE root")
    parser.add_argument("-b", "--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=200,type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument("--resultsname", default='CRNet', help="results name")
    parser.add_argument("--modelname", default='CRNet', help="results name")
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_args()
    main(opt)
