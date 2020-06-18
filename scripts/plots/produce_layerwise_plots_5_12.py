from knowledge_probing.plotting.layer_wise_plots_5_12 import produce_layer_wise_plots
from argparse import ArgumentParser


def main(args):
    print('Using order type: {}'.format(args.order_type))
    produce_layer_wise_plots(args.order_type)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--order_type', default='None',
                        choices=['None', 'naive', 'clustering'])

    args = parser.parse_args()

    main(args)
