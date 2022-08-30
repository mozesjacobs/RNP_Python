import torch

def make_model(args):
    if args.model == "RNP":
        from RNP import Net
    elif args.model =='RNP_GON':
        from RNP_GON import Net
    elif args.model =='RNP_Error1':
        from RNP_Error1 import Net
    elif args.model =='RNP_IAI':
        from RNP_IAI import Net
    elif args.model =='RNP_IAI2':
        from RNP_IAI2 import Net
    elif args.model =='IAI_Model':
        from IAI_Model import Net
    elif args.model =='IAI_Model2':
        from IAI_Model2 import Net
    return Net(args)