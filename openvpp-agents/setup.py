from setuptools import setup, find_packages


setup(
    name='openvpp-agents',
    version='0.1.0',
    author='Martin Tröschel',
    author_email='martin.troeschel@gmail.com',
    description=('The Open VPP multi-agent system'),
    # long_description=(open('README.txt').read() + '\n\n' +
    #                   open('CHANGES.txt').read() + '\n\n' +
    #                   open('AUTHORS.txt').read()),
    url='https://particon.de',
    install_requires=[
        'aiomas[mpb]>=1.0.1',
        'arrow>=0.4',
        'click>=4.0',
        'h5py>=2.5',
        'numpy>=1.8',
        'psutil>=2.2',
    ],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'openvpp-mosaik = openvpp_agents.mosaik:main',
            'openvpp-container = openvpp_agents.container:main',
        ],
    },
)
