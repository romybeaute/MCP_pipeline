__version__ = '0.1.dev'
__next_name__ = 'NICE PerMed Extension Package'

from . import germany
from . import germany_mu
from . import israel
from . import italy
from . import usanihon
from . import data


from . import equipments

equipments.register()
