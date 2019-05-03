import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GraphPool arguments.')
    io_parser = parser.add_mutually_exclusive_group()
    io_parser.add_argument('--bmname', dest='bmname',
            help='Name of the benchmark dataset')

    treepool_parser = parser.add_argument_group()
    treepool_parser.add_argument('--num-trees', dest='num_trees', type=int,
                        help='Number of trees in a graph')
    treepool_parser.add_argument('--depth', dest='depth', type=int,
                                 help='Depth of trees in a graph')
    treepool_parser.add_argument('--tree-attention', dest='attention', type=bool,
                                 help='user trees attention in a graph')

    parser.add_argument('--datadir', dest='datadir',
            help='Directory where benchmark is located')
    parser.add_argument('--logdir', dest='logdir',
            help='Tensorboard log directory')
    parser.add_argument('--cuda', dest='cuda',
            help='CUDA.')
    parser.add_argument('--max-nodes', dest='max_nodes', type=int,
            help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    parser.add_argument('--wL2', dest='w_l2', type=float,
                        help='l2 weight.')
    parser.add_argument('--seed', dest='seed', type=int,
                        help='random seed.')
    parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')
    parser.add_argument('--batch-size', dest='batch_size', type=int,
            help='Batch size.')
    parser.add_argument('--epochs', dest='num_epochs', type=int,
            help='Number of epochs to train.')
    parser.add_argument('--train-ratio', dest='train_ratio', type=float,
            help='Ratio of number of graphs training set to all graphs.')
    parser.add_argument('--num_workers', dest='num_workers', type=int,
            help='Number of workers to load data.')
    parser.add_argument('--feature', dest='feature_type',
            help='Feature used for encoder. Can be: id, deg')
    parser.add_argument('--input-dim', dest='input_dim', type=int,
            help='Input feature dimension')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
            help='Hidden dimension')
    parser.add_argument('--output-dim', dest='output_dim', type=int,
            help='Output dimension')

    parser.add_argument('--num-classes', dest='num_classes', type=int,
            help='Number of label classes')
    parser.add_argument('--alpha', dest='alpha', type=float,
                        help='alpha for leakly_relu.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
            const=False, default=True,
            help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')
    parser.add_argument('--fold-id', dest='fold_id', type=int,
            help='fold id')
    parser.add_argument('--nobias', dest='bias', action='store_const',
            const=False, default=True,
            help='Whether to add bias. Default to True.')

    parser.add_argument('--method', dest='method',
            help='Method. Possible values: base, base-set2set, soft-assign')
    parser.add_argument('--name-suffix', dest='name_suffix',
            help='suffix added to the output filename')
    parser.add_argument('--out-file', dest='out_file', type=str,
                        help='output log')
    parser.add_argument('--file-name', dest='file_name', type=str,
                        help='output log')
    parser.add_argument('--concat', dest='concat', type=bool,
                        help='concat.')
    parser.add_argument('--tree-tranfer', dest='tree_trans', type=bool,
                        help='if trans in tree.')
    parser.set_defaults(datadir='data',
                        logdir='log',
                        concat=False,
                        alpha=0.002,
                        num_trees = 8,
                        depth = 4,
                        attention = False,
                        out_file="val.txt",
                        bmname='DD',
                        max_nodes=1000,
                        cuda='0',
                        feature_type='default',
                        lr=0.000001,
                        w_l2=0.0,
                        dropout=0.0,
                        fold_id=0,
                        seed=5,
                        clip=2.0,
                        batch_size=16,
                        num_epochs=200,
                        train_ratio=0.9,
                        test_ratio=0.2,
                        num_workers=8,
                        input_dim=10,
                        hidden_dim=64,
                        output_dim=64,
                        num_classes=2,
                        num_gc_layers=2,
                        method='tree',
                        name_suffix='',
                        num_pool=1,
                        file_name='debug0',
                        tree_trans=False
                       )
    return parser.parse_args()
