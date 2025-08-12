import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from dinov2.data.datasets import EntityResourceForSimCLR
from dinov2.models.vision_transformer import vit_giant2_with_mlp
from dinov2.train.simclr import SimCLR
from dinov2.utils.config import setup_for_simclr
import dinov2.utils.utils as dinov2_utils


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 MLP SimCLR training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="", type=str, help="Output directory to save logs and checkpoints")
    parser.add_argument(
        "opts",
        help="for reuse setup() only".strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--pretrain_backbone_weight', default='',
                        help='path to pretrained backbone weights')
    parser.add_argument('--root', default='./datasets',
                        help='path to dataset')
    parser.add_argument('--hdfs_loading', default='False', type=str,
                        help='whether to load data from hdfs')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--fp16-precision', action='store_true',
                        help='Whether or not to use 16-bit precision GPU training.')

    parser.add_argument('--out_dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--log-every-n-steps', default=100, type=int,
                        help='Log every n steps')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--n-views', default=2, type=int, metavar='N',
                        help='Number of views for contrastive learning training.')
    parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
    parser.add_argument('--crop_size', default=32, type=int, help='image crop size')
    parser.add_argument('--n_view', default=2, type=int, help='number of view for each image')
    return parser


def main():
    args = get_args_parser(add_help=True).parse_args()
    cfg = setup_for_simclr(args)
    vit_config = cfg.student
    vit_kwargs = dict(
        img_size=cfg.crops.global_crops_size,
        patch_size=vit_config.patch_size,
        init_values=vit_config.layerscale,
        ffn_layer=vit_config.ffn_layer,
        block_chunks=vit_config.block_chunks,
        qkv_bias=vit_config.qkv_bias,
        proj_bias=vit_config.proj_bias,
        ffn_bias=vit_config.ffn_bias,
        num_register_tokens=vit_config.num_register_tokens,
        interpolate_offset=vit_config.interpolate_offset,
        interpolate_antialias=vit_config.interpolate_antialias,
    )
    mlp_kwargs = dict(
        out_dim=args.out_dim
    )

    # model = DinoVisionTransformerWithMLP(**vit_kwargs, **mlp_kwargs).to(torch.device("cuda"))
    model = vit_giant2_with_mlp(**vit_kwargs, **mlp_kwargs).to(torch.device("cuda"))
    if args.pretrain_backbone_weight:
        dinov2_utils.load_pretrained_weights(model.backbone, args.pretrain_backbone_weight, "teacher")


    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    train_dataset = EntityResourceForSimCLR(
        root=args.root,
        split=EntityResourceForSimCLR.Split.TRAIN,
        hdfs_load=True if args.hdfs_loading.lower() == "true" else False,
        crop_size=args.crop_size,
        n_views=args.n_views,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # freeze backbone, only train mlp layer
    optimizer = torch.optim.Adam(model.mlp.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)


if __name__ == "__main__":
    main()