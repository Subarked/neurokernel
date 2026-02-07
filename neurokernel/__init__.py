from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

from os import path
__path__.append(path.join(__path__[0], 'tools'))
