from train import train_model, make_data
from model import *
from options import args

if args.data_set_name in ['HanChuan','Houston']:
    split_image = True
else:
    split_image = False

make_data(args.data_set_name, args.data_set_path)
model = build_model(args)

if __name__ == '__main__':
    train_model(model, split_image, args)