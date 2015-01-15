"""
Merge distributed analysis sets from a FileHandler.

Usage:
    merge.py <base_path> [--cleanup]

Options:
    --cleanup   Delete distributed files after merging

"""

if __name__ == "__main__":

    from docopt import docopt
    from dedalus2.tools import logging
    from dedalus2.tools import post

    args = docopt(__doc__)
    post.merge_analysis(args['<base_path>'], cleanup=args['--cleanup'])

