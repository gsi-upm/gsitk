from setuptools import setup

VERSION = '0.1.8.1'


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    with open(filename, 'r') as f:
        lineiter = list(line.strip() for line in f)
    return [line for line in lineiter if line and not line.startswith("#")]


install_reqs = parse_requirements("requirements.txt")


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
