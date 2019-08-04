from scripts import train, test_convert_eval


def parse_test_args():
    test_parser = test_convert_eval.get_test_parser_without_logdir()
    return test_parser.parse_known_args()[0]


def main():
    test_args = parse_test_args()
    train_logdir = train.main()
    test_convert_eval.main(train_logdir, **test_args.__dict__)


if __name__ == '__main__':
    main()
