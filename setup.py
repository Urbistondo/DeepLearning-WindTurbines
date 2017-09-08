from setuptools import setup, find_packages


'''Necessary setup file to run job in Google Cloud Platform'''


setup(
    name = 'Peak model',
    version = '1.0',
    packages = find_packages(),
    description = 'Running peak model on GCP',
    author = 'Javier Urbistondo',
    author_email = 'jurbistondo.nem@gmail.com',
    install_requires = [
      'keras',
      'h5py'
    ],
    zip_safe = False
)
