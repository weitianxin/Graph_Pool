import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='TreeAttentionNetworks.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset',
            help='Input dataset.')
    benchmark_parser = io_parser.add_argument_group()
    benchmark_parser.add_argument('--bmname', dest='bmname',
            help='Name of the benchmark dataset')


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


    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int,
            help='Hidden dimension')

    parser.add_argument('--num-classes', dest='num_classes', type=int,
            help='Number of label classes')
    parser.add_argument('--num-gcn-layers', dest='num_gcn_layers', type=int,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--nobn', dest='bn', action='store_const',
            const=False, default=True,
            help='Whether batch normalization is used')
    parser.add_argument('--dropout', dest='dropout', type=float,
            help='Dropout rate.')
    parser.add_argument('--nobias', dest='bias', action='store_const',
            const=False, default=True,
            help='Whether to add bias. Default to True.')

    parser.add_argument('--pooling-method', dest='method',
            help='Method. Possible values: sum, attention')

    parser.set_defaults(datadir='data',
                        logdir='log',
                        max_nodes=10000,
                        cuda='1',
                        feature_type='default',
                        lr=0.00005,
                        w_l2=0.0, #在0.01到0.001之间选取
                        clip=2.0,
                        batch_size=16,
                        num_epochs=200,
                        train_ratio=0.75,
                        num_workers=0,
                        hidden_dim=64,
                        num_classes=2,
                        num_gcn_layers=3,
                        dropout=0.5,
                        method='others'
                       )
    return parser.parse_args()