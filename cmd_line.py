import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Template")

    # save params
    parser.add_argument("-SN", "--session_name", default="may24", type=str, help="Session name")
    parser.add_argument("-M", "--model", default="RNP", type=str, help="Which model to load (RNP, RNP_GON)")
    parser.add_argument("-EF", "--exp_folder", default="experiments", type=str, help="Folder for experiments")
    parser.add_argument('-NF', '--net_folder', default="trained_models", type=str, help="Folder for trained models")
    parser.add_argument('-TBP', '--tensorboard_folder', default="tb_runs", type=str, help="Folder for tensorboard logs")

    # model parameters
    parser.add_argument('-Z', '--z_dim', default=32, type=int, help="Dimension of r")
    parser.add_argument('-A', '--a_dim', default=6, type=int, help="Dimension of the action (theta)")
    parser.add_argument('-EHD', '--encoder_hidden_dim', default=128, type=int)
    parser.add_argument('-HHD', '--hypernet_hidden_dim', default=128, type=int)
    parser.add_argument('-ED', '--e_dim', default=64, type=int)
    parser.add_argument('-TH', '--theta', default=4, type=int)

    # learning params
    parser.add_argument('-LR', '--learning_rate', default=4e-5, type=float, help="Learning rate")
    parser.add_argument('-GF', '--gamma_factor', default=1.0, type=float, help="Learning rate decay factor (leave as 1.0 for no decay)")

    # training / testing
    parser.add_argument('-D', '--device', default=2, type=int, help="Which device to use")
    parser.add_argument('-E', '--epochs', default=20, type=int, help="Number of Training Epochs")
    parser.add_argument('-B', '--batch_size', default=50, type=int, help="Batch size")    
    parser.add_argument('-SI', '--save_interval', default=1, type=int, help="How often (in epochs) to save a checkpoint in training")
    parser.add_argument('-TI', '--test_interval', default=1, type=int, help="How often (in epochs) to evaluate the model on test data during training")
    parser.add_argument('-LCP', '--load_cp', default=0, type=int, help="Load from saved checkpoint")
    parser.add_argument('-GC', '--clip', default=None, type=int, help="Gradient clip norm value (leave as None for no clipping)")

    # data
    parser.add_argument('-ID', '--img_dim', default=28, type=int, help="Dimensions of the frames")
    parser.add_argument('-CH', '--channels', default=1, type=int, help="Number of channels in image frame")

    # other
    parser.add_argument('-PF', '--print_folder', default=1, type=int, help="Print the name of the folders things are saved in")
    
    return parser.parse_args() 