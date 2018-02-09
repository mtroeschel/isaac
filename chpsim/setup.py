from setuptools import setup, find_packages


setup(
    name='chpsim',
    version='1.0.0',
    author='Martin Tröschel',
    author_email='martin.troeschel@gmail.com',
    description=(''),
    # long_description=(open('README.txt').read() + '\n\n' +
    #                   open('CHANGES.txt').read() + '\n\n' +
    #                   open('AUTHORS.txt').read()),
    url='',
    install_requires=[
        'arrow>=0.5.4',
        'mosaik-api>=2.1',
        'numpy>=1.8',
    ],
    packages=find_packages(exclude=['tests*']),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'mosaik-chpsim = chpsim.mosaik:main',
        ],
    },
)
