from My_func import *
import optparse


def main():
    description = """
    This program is designed for training and prediction for solar panels power generation.
    """
    optp = optparse.OptionParser(description=description)
    optp.add_option('-s', '--station', default='awt')  # choose in [awt,gy,ps,lf,wtf,yms,ml]
    # station could be abbreviation of Chinese Words, such as 'awt'，
    # or could be Chinese name '阿瓦提'(not supportd yet)
    optp.add_option('-t', '--type', default='train')  # train or test
    optp.add_option('-l', '--length', default='short')  # short or super-short
    optp.add_option('-p', '--predicate', default='forecas.csv')
    optp.add_option('-n', '--file_name', default='output.csv')

    options, args = optp.parse_args()

    if options.type == 'train':
        my_train_func(options.station)
    elif options.type == 'test':
        if options.length == 'short':
            my_spredict_func(options.predicate,options.file_name)
        elif options.length == 'super-short':
            my_sspredict_func(options.predicate, options.file_name)


if __name__ == '__main__':
    main()
