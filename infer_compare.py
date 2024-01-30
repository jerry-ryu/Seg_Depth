import argparse
import os
import sys
import copy

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import model_io
from models import UnetAdaptiveBins
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from dataloader import DepthDataLoader
from utils import RunningAverageDict
import pandas as pd


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)

def viz(impath, img, final, gt, validmask, bin_np, metric):
    
    if args.save_dir is not None:
        if args.dataset == 'nyu':
            if impath[0] == '/':
                impath = impath[1:]
            factor = 1000.
            norm_vmin = 0
            norm_vmax = 5000
            if args.dataset_kind == 'test':
                error_vmin = -2000
                error_vmax = 2000
            else:
                error_vmin = -1000
                error_vmax = 1000
        elif args.dataset == 'sun':
            if impath[0] == '/':
                impath = impath[1:]
            factor = 10000.
            norm_vmin = 0
            norm_vmax = 50000
            if args.dataset_kind == 'test':
                error_vmin = -2000
                error_vmax = 2000
            else:
                error_vmin = -1000
                error_vmax = 1000
        else:
            dpath = impath.split('/')
            impath = dpath[1] + "_" + dpath[-1]
            impath = impath.split('.')[0]
            factor = 256.
            norm_vmin = 0
            norm_vmax = 20000
            if args.dataset_kind == 'test':
                error_vmin = -1000
                error_vmax = 1000
            else:
                error_vmin = -500
                error_vmax = 500
        
        pred_path = os.path.join(args.save_dir, impath)
        pred = (final * factor).astype('uint16')

        img = img.squeeze()
        img = img.transpose((1,2,0))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        denormalized_image = img * std + mean
        img = denormalized_image.clip(0, 255)
        #pred_map
        pred_cmap = plt.get_cmap('plasma')
        pred_norm = Normalize(vmin=norm_vmin, vmax=norm_vmax)
        
        
        #gt_map
        valid_map = np.ones_like(gt)
        valid_map[validmask==False] = np.NaN
        if args.dataset == 'nyu':
            gt_map = (gt*factor)*valid_map
        else:
            gt_map = (gt*factor)
        gt_cmap = plt.get_cmap('plasma')
        gt_cmap.set_bad(color="Black")
        gt_norm = Normalize(vmin=norm_vmin, vmax=norm_vmax)
        
        # error map
        error_cmap = plt.get_cmap("RdBu")
        error_norm = Normalize(vmin=error_vmin, vmax=error_vmax)
        
        if args.dataset == 'nyu':
            error_cmap.set_bad(color="Black")
            error_map = (pred - gt*factor)*valid_map
        else:
            error_cmap.set_bad(color="White", alpha = 1.0)
            error_map = (pred - gt*factor)*valid_map
        
        # bins
        flt_gt = gt[validmask].flatten()
        flt_pred = pred[validmask].flatten()/factor
        bin_hist = bin_np[0].tolist()
        bins_uni = np.linspace(args.min_depth, args.max_depth, num = args.n_bins).tolist()
        bins_uni_20 = np.linspace(args.min_depth, args.max_depth, num = 20).tolist()
        
        #chamfer_distance
        from pytorch3d.loss import chamfer_distance
        from torch.nn.utils.rnn import pad_sequence
        bin_centers = 0.5 * (bin_np[:, 1:] + bin_np[:, :-1])
        input_points = np.expand_dims(bin_centers,2)  # .shape = n, p, 1
        input_points = torch.from_numpy(input_points)
        # n, c, h, w = target_depth_maps.shape
        
        target_depth_maps = torch.from_numpy(gt)
        target_depth_maps = target_depth_maps.unsqueeze(0)
        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long()
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        cham_dist, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        
        # metric
        metric_dict = copy.deepcopy(metric)
        metric_dict["cham_dist"] = cham_dist.item()
        metric_dict["path"] = pred_path
        
        # plot
        if args.dataset == 'kitti':
            fig = plt.figure(figsize= (50,28))
        elif args.dataset == 'sun':
            fig = plt.figure(figsize= (15,18))
        else:
            fig = plt.figure(figsize= (15,14))
        
        
        # RGB
        plt.subplot(4, 3, 1)
        plt.imshow((img * 255).astype(np.uint8))
        plt.title("RGB image")

        # pred_norm
        plt.subplot(4, 3, 2)
        plt.imshow(pred, cmap=pred_cmap, norm=pred_norm)
        plt.colorbar()
        plt.title(f"Pred depth_norm({norm_vmin},{norm_vmax})")

        # GT_norm
        plt.subplot(4, 3, 3)
        plt.imshow(gt_map, cmap=gt_cmap, norm=gt_norm)
        plt.colorbar()
        plt.title(f"Ground Truth_norm({norm_vmin},{norm_vmax})")

        # Error_map
        plt.subplot(4, 3, 4)
        plt.imshow(error_map, cmap=error_cmap, norm=error_norm)
        plt.colorbar()
        plt.title("Error map (pred - gt)")
        plt.text

        # pred
        plt.subplot(4, 3, 5)
        plt.imshow(pred, cmap=pred_cmap)
        plt.colorbar()
        plt.title("Pred depth")

        # GT
        plt.subplot(4, 3, 6)
        plt.imshow(gt_map, cmap=gt_cmap)
        plt.colorbar()
        plt.title("Ground Truth")
        
        # pred_bins hist
        plt.subplot(4, 3, 7)
        n, bins_stand, patches = plt.hist(flt_gt, bins = bin_hist, rwidth = 0.95)
        cm = plt.get_cmap("viridis")
        min_value = np.min(bins_stand)
        max_value = np.max(bins_stand)
        min_max_stand = max_value - min_value
        bins_nom = [x / min_max_stand for x in bins_stand]
        for c, p in zip(bins_nom, patches):
            plt.setp(p, 'facecolor', cm(c))
        plt.title("GT hist. with Adabins")

        # uniform_bins hist n_bins
        plt.subplot(4, 3, 8)
        n, bins, patches = plt.hist(flt_gt, bins = bins_uni_20, rwidth = 0.95)
        bins_nom = [x / min_max_stand for x in bins]
        for c, p in zip(bins_nom, patches):
            plt.setp(p, 'facecolor', cm(c))
            plt.xlim(min_value, max_value)
        plt.title("GT hist. with uniform bins(20)")
        plt.xlim(0.,10.)
        
        # bins hist
        plt.subplot(4, 3, 9)
        n, bins, patches = plt.hist(bin_hist, bins=20)
        bins_nom = [x / min_max_stand for x in bins]
        for c, p in zip(bins_nom, patches):
            plt.setp(p, 'facecolor', cm(c))
        plt.title("Hist. of  Adabins")
        
        # pred_bins hist
        plt.subplot(4, 3, 10)
        n, bins_stand, patches = plt.hist(flt_pred, bins = bin_hist, rwidth = 0.95)
        cm = plt.get_cmap("viridis")
        bins_nom = [x / min_max_stand for x in bins_stand]
        for c, p in zip(bins_nom, patches):
            plt.setp(p, 'facecolor', cm(c))
        plt.title("Pred hist. with Adabins")
        
        # uniform_bins hist n_bins
        plt.subplot(4, 3, 11)
        n, bins, patches = plt.hist(flt_gt, bins = bins_uni, rwidth = 0.95)
        bins_nom = [x / min_max_stand for x in bins]
        for c, p in zip(bins_nom, patches):
            plt.setp(p, 'facecolor', cm(c))
            plt.xlim(min_value, max_value)
        plt.title(f"GT hist. with uniform bins({args.n_bins})")
        plt.xlim(0.,10.)
        
        
        
        # metrics
        plt.subplot(4, 3, 12)
        plt.text(0.1,0.0,
                 f"a1: {metric_dict['a1']}\na2: {metric_dict['a2']}\na3: {metric_dict['a3']}\
                 \nrmse_log: {metric_dict['rmse_log']}\nrmse: {metric_dict['rmse']}\
                 \nlog_10: {metric_dict['log_10']}\nabs_rel: {metric_dict['abs_rel']}\nsilog: {metric_dict['silog']}\
                 \nsq_rel: {metric_dict['sq_rel']}\ncham_dist: {metric_dict['cham_dist']}", fontsize=16)
        plt.axis("off")
        
        # save
        if not os.path.exists(os.path.dirname(pred_path)):
            os.makedirs(os.path.dirname(pred_path))
        plt.savefig(pred_path)
        plt.close()
        
        if args.dataset == 'sun':
            fig = plt.figure(figsize= (8,8))
        
            # pred freq.
            plt.subplot(2, 2, 1)
            pred_fft = np.fft.fft2(pred)
            pred_fft = np.fft.ifftshift(pred_fft)
            pred_fft_log = 20*np.log(np.abs(pred_fft))
            plt.imshow(pred_fft_log)
            plt.colorbar()
            plt.title("Pred freq.")

            # pred freq.
            plt.subplot(2, 2, 2)
            gt_fft = np.fft.fft2(gt_map)
            gt_fft = np.fft.ifftshift(gt_fft)
            gt_fft_log = 20*np.log(np.abs(gt_fft))
            plt.imshow(gt_fft_log)
            plt.colorbar()
            plt.title("GT freq.")
            
            # freq error log
            plt.subplot(2, 2, 3)
            plt.imshow(gt_fft_log - pred_fft_log)
            plt.colorbar()
            plt.title("freq error (log, gt - pred)")
            
            # freq error log
            plt.subplot(2, 2, 4)
            plt.imshow(np.abs(gt_fft) - np.abs(pred_fft))
            plt.colorbar()
            plt.title("freq error (gt - pred)")
            
            gt_fft_path = pred_path.replace(".jpg", "_gt_freq.jpg")
            plt.savefig(gt_fft_path)
            plt.close()
            
        
    return metric_dict
        
        

# def denormalize(x, device='cpu'):
#     mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
#     std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
#     return x * std + mean
#
def predict_tta(model, image, args):
    bins, pred = model(image)
    #     pred = utils.depth_norm(pred)
    #     pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)
    #     pred = np.clip(pred.cpu().numpy(), 10, 1000)/100.
    pred = np.clip(pred.cpu().numpy(), args.min_depth, args.max_depth)

    image = torch.Tensor(np.array(image.cpu().numpy())[..., ::-1].copy()).to(device)

    pred_lr = model(image)[-1]
    #     pred_lr = utils.depth_norm(pred_lr)
    #     pred_lr = nn.functional.interpolate(pred_lr, depth.shape[-2:], mode='bilinear', align_corners=True)
    #     pred_lr = np.clip(pred_lr.cpu().numpy()[...,::-1], 10, 1000)/100.
    pred_lr = np.clip(pred_lr.cpu().numpy()[..., ::-1], args.min_depth, args.max_depth)
    final = 0.5 * (pred + pred_lr)
    final = nn.functional.interpolate(torch.Tensor(final), image.shape[-2:], mode='bilinear', align_corners=True)
    return torch.Tensor(final), bins


def eval(model, test_loader, args, gpus=None, ):
    if gpus is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = gpus[0]

    if args.save_dir is not None and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    df = pd.DataFrame()
    
    metrics = RunningAverageDict()
    # crop_size = (471 - 45, 601 - 41)
    # bins = utils.get_bins(100)
    with torch.no_grad():
        model.eval()

        sequential = test_loader
        for batch in tqdm(sequential):
            batch_size = len(batch['focal'])
            image = batch['image'].to(device)
            gt = batch['depth'].to(device)
            final, bins = predict_tta(model, image, args)
            final = final.squeeze().cpu().numpy()

            # final[final < args.min_depth] = args.min_depth
            # final[final > args.max_depth] = args.max_depth
            final[np.isinf(final)] = args.max_depth
            final[np.isnan(final)] = args.min_depth
            
            gt = gt.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt > args.min_depth, gt < args.max_depth)

            if args.garg_crop or args.eigen_crop:
                _, gt_height, gt_width = gt.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[:,int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[:,int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                        int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    else:
                        eval_mask[:,45:471, 41:601] = 1
                        
                valid_mask = np.logical_and(valid_mask, eval_mask)
            #             gt = gt[valid_mask]
            #             final = final[valid_mask]
            image_np = image.cpu().numpy()
            bins_np = bins.cpu().numpy()
            gt_s = np.split(gt, batch_size, axis = 0)
            final_s = np.split(final, batch_size, axis = 0)
            valid_mask_s = np.split(valid_mask, batch_size, axis = 0)
            image_s = np.split(image_np, batch_size, axis = 0)
            bins_s = np.split(bins_np, batch_size, axis = 0)
            
            for gt_i, final_i, valid_mask_i, image_i, bins_i, path in zip(gt_s, final_s, valid_mask_s, image_s, bins_s, batch["image_path"]):
                metric = compute_errors(gt_i[valid_mask_i], final_i[valid_mask_i])
                metrics.update(metric)
                metric_dict = viz(path, image_i, final_i.squeeze(), gt_i.squeeze(), valid_mask_i.squeeze(), bins_i, metric)
                df = pd.concat([df, pd.DataFrame.from_dict([metric_dict])])
                
    metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
    print(f"Metrics: {metrics}")
    f = open(os.path.join(args.save_dir, "metric.txt"), 'w')
    print(f"Metrics: {metrics}", file=f)
    f.close()
    
    df.to_csv(os.path.join(args.save_dir, "metric.csv"), index=False)


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Model evaluator', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--n-bins', '--n_bins', default=256, type=int,
                        help='number of bins/buckets to divide depth range into')
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    parser.add_argument('--save-dir', '--save_dir', default=None, type=str, help='Store predictions in folder')
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")

    parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")

    parser.add_argument("--data_path", default='../dataset/nyu/sync/', type=str,
                        help="path to dataset")
    parser.add_argument("--gt_path", default='../dataset/nyu/sync/', type=str,
                        help="path to dataset gt")

    parser.add_argument('--filenames_file',
                        default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
                        type=str, help='path to the filenames text file')

    parser.add_argument('--input_height', type=int, help='input height', default=416)
    parser.add_argument('--input_width', type=int, help='input width', default=544)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)

    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')

    parser.add_argument('--data_path_eval',
                        default="../dataset/nyu/official_splits/test/",
                        type=str, help='path to the data for online evaluation')
    parser.add_argument('--gt_path_eval', default="../dataset/nyu/official_splits/test/",
                        type=str, help='path to the groundtruth data for online evaluation')
    parser.add_argument('--filenames_file_eval',
                        default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
                        type=str, help='path to the filenames text file for online evaluation')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help="checkpoint file to use for prediction")

    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('--eigen_crop', help='if set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
    parser.add_argument('--do_kb_crop', help='Use kitti benchmark cropping', action='store_true')
    parser.add_argument('--viz', help='vizualization config', action='store_true')
    parser.add_argument("--eval_dataset", default='nyu', type=str, help="Dataset to eval on")
    parser.add_argument('--bs', default=16, type=int, help='batch size')
    parser.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")
    parser.add_argument("--dataset_kind", default=11, type=str, help="dataset category for EDA")

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    args.batch_size = args.bs
    args.num_threads = args.workers
    
    # args = parser.parse_args()
    args.gpu = int(args.gpu) if args.gpu is not None else 0
    args.distributed = False
    device = torch.device('cuda:{}'.format(args.gpu))
    test = DepthDataLoader(args, 'online_eval').data
    model = UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                   norm='linear').to(device)
    model = model_io.load_checkpoint(args.checkpoint_path, model)[0]
    model = model.eval()

    eval(model, test, args, gpus=[device])
