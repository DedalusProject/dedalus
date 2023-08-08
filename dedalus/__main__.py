"""
Dedalus module interface.

Usage:
    dedalus test [--report]
    dedalus bench
    dedalus cov
    dedalus get_config
    dedalus get_examples
    dedalus merge_sets <set_paths>... [--cleanup] [--joint_path=<joint_path>]

Options:
    --joint_path=<joint_path> optional name for merged sets [default: None]
"""

if __name__ == "__main__":

    import sys
    import pathlib
    import shutil
    import tarfile
    from docopt import docopt
    from dedalus.tools import logging, post
    from dedalus.tests import test, bench, cov

    args = docopt(__doc__)
    if args['test']:
        sys.exit(test(report=args['--report']))
    elif args['bench']:
        sys.exit(bench())
    elif args['cov']:
        sys.exit(cov())
    elif args['get_config']:
        config_path = pathlib.Path(__file__).parent.joinpath('dedalus.cfg')
        shutil.copy(str(config_path), '.')
    elif args['get_examples']:
        example_path = pathlib.Path(__file__).parent.joinpath('examples.tar.gz')
        with tarfile.open(str(example_path), mode='r:gz') as archive:
            archive.extractall('dedalus_examples')
    elif args['merge_sets']:
        joint_path = args['--joint_path']
        if joint_path == 'None':
            joint_path = None
        post.merge_sets(args['<set_paths>'], joint_path=joint_path, cleanup=args['--cleanup'])
        

