import json
import argparse
from trainer import train

def main():
    args = setup_parser().parse_args()

    param = load_json(args.config)
    args = vars(args) 
        
    param.update(args)  
    print(param)
    print("start!")

    train(param)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of APART.')
    parser.add_argument('--config', type=str, default='./exps/apart_cifar_shuffle.json', 
                        help='Json file of settings.')
    parser.add_argument('--text', type=str, default='', 
                        help='Description of this run.')
    return parser

if __name__ == '__main__':
    main()

