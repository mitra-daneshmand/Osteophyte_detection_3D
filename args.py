import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Parse model training options')
    parser.add_argument('-net', '--network', default='ResNet',
                        help='Network type. Options: ResNet (default), ResNet, GCN)')

    parser.add_argument('-d', '--model_depth', default=10,
                        help='Model depth. Options: 10 (default), 18, 34)')

    parser.add_argument('-out', '--output_dir', default='sessions/',
                        help='Complete path for the model weights and other results. (default: sessions/)')

    parser.add_argument('--pretrain_path', default='pretrained_models/resnet_10_23dataset.pth',
                        help='Pretrained model (.pth)')  # pretrained_models/resnet_10_23dataset.pth

    parser.add_argument('-csv', '--csv_dir', default='../data/CSVs/',
                        help='CSV files directory: ../data/CSVs/ (default)')

    parser.add_argument('--n_classes', default=1, type=int,
                        help='Number of classes')

    parser.add_argument('--ft_portion', default='complete', type=str,
                        help='The portion of the model to apply fine tuning, Options: complete, last_layer, except_first_layer, 6_layers')

    parser.add_argument('-e', '--epochs', default=30, type=int,
                        help='Number of epochs to train the model. (default: 20)')

    parser.add_argument('-bs', '--batch_size', default=16, type=int,
                        help='Training batch size. (default: 16)')

    parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float,
                        help='Learning rate. (default: 1e-3)')

    parser.add_argument('-wd', '--weight_decay', default=1e-5,
                        help='Weight decay for training. Options: 1e-5 (default), 1e-4')

    parser.add_argument('-augmentation', '--aug', default=True,
                        help='Doing augmentation or not. Options: True (default), False')

    parser.add_argument('-tar_comp', '--target_comp', default='FL',
                        help='Knee area for OST progression definition. Options: All (default), FM, TL, TM, All, b_multi_label, m_multi_label, fltl, fmtm')

    parser.add_argument('-t', '--tissue', default='Bones',
                        help='Knee tissue for training. Options: All (default), Bones, Cartilages, BoneMenisci, BoneCartilage, MenisciCartilage')

    parser.add_argument('-lm', '--lm', default='lateral',
                        help='lateral or medial comps. Options: ''(default), medial, lateral')

    args = parser.parse_args()
    return args
