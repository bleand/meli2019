import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--clean-tset', action='store_true')
    parser.add_argument('--create-tset', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--calc-labels', action='store_true')
    parser.add_argument('--fresh-start', action='store_true')
    parser.add_argument('--multi-gpu', action='store_true')

    args = parser.parse_args()
    print(args)


if __name__ == '__main__':
    main()