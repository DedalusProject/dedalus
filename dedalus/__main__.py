"""
Dedalus module interface.

Usage:
    dedalus test
    dedalus bench
    dedalus cov
    dedalus get_config
    dedalus get_examples
    dedalus merge_procs <base_path> [--cleanup]
    dedalus merge_sets <joint_path> <set_paths>... [--cleanup]

Options:
    --cleanup   Delete distributed files after merging

"""

if __name__ == "__main__":

    import pathlib
    import shutil
    import tarfile
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tests import test, bench, cov

    args = docopt(__doc__)
    if args['test']:
        test()
    elif args['bench']:
        bench()
    elif args['cov']:
        cov()
    elif args['get_config']:
        config_path = pathlib.Path(__file__).parent.joinpath('dedalus.cfg')
        shutil.copy(str(config_path), '.')
    elif args['get_examples']:
        example_path = pathlib.Path(__file__).parent.joinpath('examples.tar.gz')
        with tarfile.open(str(example_path), mode='r:gz') as archive:
            archive.extractall('dedalus_examples')
    elif args['merge_procs']:
        post.merge_process_files(args['<base_path>'], cleanup=args['--cleanup'])
    elif args['merge_sets']:
        post.merge_sets(args['<joint_path>'], args['<set_paths>'], cleanup=args['--cleanup'])

