import pip
#from distutils.core import setup
from setuptools import setup
from pip.req import parse_requirements

VERSION = '0.1.6'


try:
    install_reqs = parse_requirements(
        "requirements.txt", session=pip.download.PipSession())
except AttributeError:
    install_reqs = parse_requirements("requirements.txt")

install_reqs = [str(ir.req) for ir in install_reqs]


setup(name = 'gsitk',
      packages = ['gsitk'], # this must be the same as the name above
      version = VERSION,
      description = 'gsitk is a library on top of scikit-learn that eases the development process on NLP machine learning driven projects.',
      author = 'Oscar Araque',
      author_email = 'oscar.aiborra@gmail.com',
      url = 'https://github.com/gsi-upm/gsitk', # URL to the github repo
      download_url =
      'https://github.com/gsi-upm/gsitk/tarball/{}'.format(VERSION),
      keywords = ['sentiment analysis', 'nlp', 'machine learning'], # list of keywords that represent your package
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      classifiers = [],
      install_requires=install_reqs,
      include_package_data=True
)
