import torch

def make_model(args):
    if args.model == "RNP":
        from RNP import Net
    elif args.model =='RNP_GON':
        from RNP_GON import Net
    return Net(args)