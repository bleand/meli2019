from pipeline import pipeline
import argparse
import logging
import sys


logging.getLogger('gensim').setLevel(logging.FATAL)

logger = logging.getLogger('root')
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s")
logger.setLevel('DEBUG')
logger.error(f"Setting log level to {logger.level}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--clean-tset', action='store_true')
    parser.add_argument('--create-tset', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--calc-labels', action='store_true')
    parser.add_argument('--fresh-start', action='store_true')
    parser.add_argument('--multi-gpu', action='store_true')
    parser.add_argument('--sample', action='store_true')

    args = parser.parse_args()
    pipeline(args)


if __name__ == '__main__':
    main()
