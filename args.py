import argparse


def build_args():
    parser = argparse.ArgumentParser(description="Model")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--load_model", action="store_true")

    # graph transformer
    parser.add_argument('--hops', type=int, default=3, help='Hop of neighbors to be calculated')
    parser.add_argument('--pe_dim', type=int, default=128, help='position embedding size')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer size')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=2, help='Number of Transformer heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout')
    parser.add_argument('--attention_dropout', type=float, default=0.1, help='Dropout in the attention layer')
    parser.add_argument("--activation", type=str, default="elu")

    # adjustable parameters
    parser.add_argument("--epoch", type=int, default=500, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--loss_fn", type=str, default="mse")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="weight decay")

    # File parameter
    parser.add_argument("--sample_name1", type=str, default="Human_Breast_Cancer_Rep1")
    parser.add_argument("--root_path1", type=str, default='./datasets/Human_Breast_Cancer_Rep1/')
    parser.add_argument("--sample_name2", type=str, default="Human_Breast_Cancer_Rep2")
    parser.add_argument("--root_path2", type=str, default='./datasets/Human_Breast_Cancer_Rep2/')
    parser.add_argument("--save", type=bool, default=True)
    parser.add_argument("--save_tag", type=str, default='fig2')
    parser.add_argument("--output_folder", type=str, default="./outputs/")

    parser.add_argument("--image_encoder", type=str, default="uni")
    parser.add_argument("--img_batch_size", type=int, default=64)
    parser.add_argument("--num_neighbors", type=int, default=7)
    parser.add_argument("--scale", type=float, default=0.363788)
    parser.add_argument("--cell_diameter", type=float, default=-1, help="By physical size (um)")
    parser.add_argument("--resolution", type=float, default=64, help="By pixels")

    # read parameters
    args = parser.parse_args()
    return args