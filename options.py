import argparse
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--data_set_path',type=str,default='../dataset')
parser.add_argument('--data_set_name', type=str, default='PaviaU')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--patch_size', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epoch', type=int, default=200)
parser.add_argument('--flag_num_ratio', type=int, default=1)  # 1: fixed number for each class, 0: ratio for each class
parser.add_argument('--train_samples', type=int, default=30)
parser.add_argument('--val_samples', type=int, default=10)
parser.add_argument('--train_ratio', type=int, default=0.1)
parser.add_argument('--val_ratio', type=int, default=0.01)
parser.add_argument('--record_computecost',type=bool,default=True)
parser.add_argument('--model',type=str,default='S4Mamba')

args = parser.parse_args()
