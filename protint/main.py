"""Central script for ProtInt, a tool for label-free proteomic data integration between patient tumours and cancer cell lines.

"""
import os
import sys
import argparse
import importlib

from protint.model_cvae import BatchCVAEFeatureWiseDropout
# Helpers to manage sys.path and model loadings
def add_path(p):
    if p and p not in sys.path:
        sys.path.insert(0, p)

def resolve_basedir(args):
    if getattr(args, "basedir", None):
        return os.path.abspath(args.basedir)
    return os.getcwd()

def import_symbol(module_name: str, symbol: str):
    mod = importlib.import_module(module_name)
    return getattr(mod, symbol)


# main function
def main():
    parser = argparse.ArgumentParser(prog='protint', 
                                     description='''ProtInt for label-free proteomic data integration between patient tumours and cancer cell lines.''')
    
    subparsers = parser.add_subparsers(dest='mode')
    
    
    ###################### Train Mode #######################
    tparser = subparsers.add_parser('train',help='Train the ProtInt model')
    
    tparser.add_argument(
        "--train_file",
        metavar = '',
        help = "A h5ad file that contains all the samples as well as a 'data_source' column. The data should have a layer called 'detected' that contains the boolean mask for detected values. The data should be log-transformed and missing values replaced by 0 before input to the model.",
    )
    
    tparser.add_argument(
        "--use_sparse_mat",
        action = "store_true",
        help = "If to use sparse dataloader. Can be used when the training dataset is huge. Default False"
    )

    # for cycle consistency
    tparser.add_argument(
        "--cyc_weight",
        type = float,
        metavar = '',
        default=0,
        help = "Weight of the backward cycle consistency loss (not used for main results). Default 0",
    )

    tparser.add_argument(
        "--dropout_weight",
        type = float,
        metavar = '',
        default = 0.1,
        help = "Weight of the probabilistic dropout loss. Default 0.1",
    )

    tparser.add_argument(
        "--train_dropout",
        action="store_true",
        help = "Whether to train the dropout parameters (rho and zeta) or not (keep fixed from the initial estimate)",
    )
    
    tparser.add_argument(
        "--src_adv_weight",
        type = float,
        metavar = '',
        default=0.01,
        help = "Weight of the source adversary loss. Default 0.01",
    )
    
    tparser.add_argument(
        "--kl_weight",
        type = float,
        default=1e-6,
        metavar = '',
        help = 'Default 1e-6. Weight for KL loss.'
    )

    tparser.add_argument(
        "--src_adv_lr",
        type = float,
        metavar = '',
        help = "Learning rate. Default 1e-3",
        default=1e-3
    )

    tparser.add_argument(
        "--batch_lr",
        type = float,
        metavar = '',
        help="Learning rate. Default 1e-3",
        default=1e-3
    )
    

    tparser.add_argument(
        "--val_set_size",
        type = float,
        metavar = '',
        help = "Fraction of samples that constitute the validation set. Default 0.0",
        default = 0.0
    )
    
    tparser.add_argument(
        "--encoding_dim",
        type=int,
        default=32,
        metavar='',
        help="Size of the latent embeddings (enc_dim). Default 32",
    )

    # New flexible architecture args
    tparser.add_argument(
        "--hidden_dims",
        type=str,
        default="128 64",
        help='Hidden layer widths as ONE string, e.g. --hidden_dims "256 128" or "256,128" or "[256, 128]"'
    )

    tparser.add_argument(
        "--activation",
        type=str,
        default="softplus",
        choices=["softplus", "relu", "selu", "leaky_relu", "elu", "gelu", "tanh", "sigmoid", "identity"],
        help="Activation function used in encoder/decoder. Default softplus."
    )

    tparser.add_argument(
        "--norm",
        type=str,
        default="batchnorm",
        choices=["layernorm", "batchnorm", "instancenorm", "groupnorm", "none"],
        help="Normalization layer used in encoder/decoder. Default layernorm."
    )

    tparser.add_argument(
        "--dropout_p",
        type=float,
        default=0.1,
        help="Dropout probability in encoder blocks. Default 0.1."
    )

    # parameters for the normalization methods
    tparser.add_argument(
    "--norm_eps",
    type=float,
    default=1e-5,
    help="Epsilon for normalization layers (LayerNorm, BatchNorm, GroupNorm, InstanceNorm)."
    )

    tparser.add_argument(
        "--bn_momentum",
        type=float,
        default=0.1,
        help="Momentum for BatchNorm / InstanceNorm. Default 0.1."
    )

    tparser.add_argument(
        "--gn_groups",
        type=int,
        default=8,
        help="Number of groups for GroupNorm. Default 8."
    )


    tparser.add_argument(
        "--balanced_sources_ae",
        action = "store_true",
        help = "Flag that enables sample weights to balance according to the source in ae loss. Default False"
    )
    
    tparser.add_argument(
        "--balanced_sources_src_adv",
        action = "store_true",
        help = "Flag that enables sample weights to balance according to the source in source adversary loss. Default False"
    )
    
    tparser.add_argument(
        "--batch_size",
        type = int,
        metavar = '',
        default = 512,
        help = 'Default 512'
    )
    
    tparser.add_argument(
        "--epochs",
        type = int,
        metavar = '',
        default = 3000,
        help = 'Max number of training epochs, Default 3000. Eearly Stopping implemented.'
    )
    tparser.add_argument(
        "--random_seed",
        type = int,
        default=100,
        metavar = '',
        help = 'Default 100'
    )
    
    
    tparser.add_argument(
        "--patience",
        type = int,
        default = 100,
        metavar = '',
        help = "Number of patience epochs for early stopping. Default 100"
    )
    
    tparser.add_argument(
        "--output_dir",
        type = str,
        default = None,
        metavar = '',
        help="Output path in case MLflow not used.'"
    )
    
    ##### MLFlow arguments####
    tparser.add_argument(
        "--use_mlflow",
        action = "store_true",
        help = "Used if all results to be tracked by MLflow. Default False"
    )
    
    tparser.add_argument(
        "--experiment_name",
        type=str,
        default = "protint",
        metavar = '',
        help ='Expriment name for MLFlow. Default protint'
    )
    
    tparser.add_argument(
        "--run_name",
        type = str,
        default = "run",
        metavar = '',
        help = "Run name for MLFlow. Default run"
    )
    
    tparser.add_argument(
        "--tmp_dir",
        type = str,
        default = "tmp",
        metavar = '',
        help = "Temporary directory for MLflow. Default ./tmp"
    )
    
    
    
    
    
    ###################### Projection Mode #######################
    pparser = subparsers.add_parser('projection',help = 'Make projection')
    

    pparser.add_argument(
        "--basedir",
        type=str,
        default="/share/home/tacongqu/projects/cellLine_patient_project",
        help="Base directory containing model/trainer modules (used to construct relative paths)."
    )
    pparser.add_argument(
        "--model_dir",
        type = str,
        metavar = '',
        help = "Path to model file",
    )
    
    pparser.add_argument(
        "--onto",
        type = str,
        metavar = '',
        help = "The target 'data_source' ID which all samples will be projected onto",
    )
    
    pparser.add_argument(
        "--projection_file",
        type = str,
        metavar = '',
        help = "An input file that contains all the gene expression of all samples. h5ad format"
    )
    
    pparser.add_argument(
        "--output_file",
        type = str,
        metavar = '',
        help = "Name for output h5ad file for projected values",
    )
    
    pparser.add_argument(
        "--decimals",
        type = int,
        default=4,
        metavar = '',
        help = "Floating-point numbers for the output file. Default 4",
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        
        from protint.train import main as train_main
        train_main(args, model_cls=BatchCVAEFeatureWiseDropout)

        
    if args.mode == "projection":

        from protint.projection import main as projection_main
        projection_main(args, model_cls=BatchCVAEFeatureWiseDropout)

    

if __name__ == "__main__": 
    main()
    
