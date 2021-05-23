"""
Configuration handling.

"""

from configparser import ConfigParser
import os


# Create config
config = ConfigParser()

# Read defaults, user, and local files
config.read(os.path.join(os.path.dirname(__file__), '../dedalus.cfg'))
config.read(os.path.expanduser('~/.dedalus/dedalus.cfg'))
config.read('dedalus.cfg')

