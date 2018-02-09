Installation
============

This document describes how to setup a development environment for ISAAC.

ISAAC requires Python 3.4 and above.  So far, it has only been tested on Linux
(mainly on Ubuntu) and OS X.  In order to install some of the
required packages (like NumPy or h5py), you may need a compiler.


Quick install
-------------

If you know how to install missing dependencies on your own have have
virtualenv(wrapper) installed, this sections provides brief installation and
setup instructions for ISAAC.

If you should run into any problems, refer to the detailed instructions below.

Clone the repositories https://bitbucket.org/particon/openvpp-agents and
https://bitbucket.org/particon/chpsim to your machine.  Create a virtualenv,
:command:`cd` into the :file:`openvpp-agents` directory and install the
requirements from :file:`requirements-setup.txt`:

.. code-block:: console

   $ hg clone https://bitbucket.org/particon/openvpp-agents
   $ hg clone https://bitbucket.org/particon/chpsim
   $ mkvirtualenv -p python3 isaac
   (isaac)$ cd openvpp-agents
   (isaac)$ pip install -e .
   (isaac)$ pip install -r requirements-setup.txt

You can now run the tests to verify that everything works fine:

.. code-block:: console

   (isaac)$ py.test
   (isaac)$ # To also run the system tests:
   (isaac)$ py.test -m1


Detailed instructions
---------------------

Linux
^^^^^

The following instructions were mainly tested on Ubuntu. For other
distributions, the set of dependencies that you need to install may vary:

.. code-block:: console

   $ sudo apt-get install python3 python3-dev python3-pip build-essential libhdf5-dev libmsgpack-dev libatlas-base-dev

Furthermore, we need virtualenv which can create isolated Python environments
for different projects.  We'll also install *virtualenvwrapper* which
simplifies your life with virtualenvs:

.. code-block:: console

   $ sudo python3 -m pip install -U pip virtualenv virtualenvwrapper
   $ # Update your bashrc to load venv. wrapper automatically:
   $ echo "# Virtualenvwrapper" >> ~/.bashrc
   $ echo "export VIRTUALENVWRAPPER_PYTHON=`which python3`" >> ~/.bashrc
   $ echo ". $(which virtualenvwrapper.sh)" >> ~/.bashrc
   $ . ~/.bashrc

Now you can create a new virtualenv, ``cd`` into the project directory (the one
containing *this* file), and install all requirements:

.. code-block:: console

   $ hg clone https://bitbucket.org/particon/openvpp-agents
   $ hg clone https://bitbucket.org/particon/chpsim
   $ mkvirtualenv -p python3 isaac
   (isaac)$ cd openvpp-agents
   (isaac)$ export HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial/
   (isaac)$ pip install -e .
   (isaac)$ pip install -r requirements-setup.txt

.. note::

   Exporting the environment variable *HDF5_DIR* may not be necessary in all
   cases (e.g., if you use Ubuntu 14.04), but it also does not hurt.

Now, you should be able to run the test suite:

.. code-block:: console

   (isaac)$ py.test

See :command:`py.test --help` for more information.


OS X
^^^^

OS X ships with an outdated version of Python.  The best/easiest way to install
Python 3.5 and other dependencies is to use Homebrew_.  Open a terminal window
and run the following command:

.. _Homebrew: http://brew.sh/

.. code-block:: console

   $ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Once the installation is successful, you can install Python 3 and the build
dependencies:

.. code-block:: console

   $ brew install python3 hdf5 msgpack

Furthermore, we need virtualenv which can create isolated Python environments
for different projects.  We'll also install *virtualenvwrapper* which
simplifies your life with virtualenvs:

.. code-block:: console

   $ python3 -m pip install -U virtualenv virtualenvwrapper
   $ # Update your bashrc to load venv. wrapper automatically:
   $ echo "# Virtualenvwrapper" >> ~/.bashrc
   $ echo "export VIRTUALENVWRAPPER_PYTHON=`which python3`" >> ~/.bashrc
   $ echo ". $(which virtualenvwrapper.sh)" >> ~/.bashrc
   $ . ~/.bashrc

Now you can create a new virtualenv, ``cd`` into the project directory (the one
containing *this* file), and install all requirements:

.. code-block:: console

   $ hg clone https://bitbucket.org/particon/openvpp-agents
   $ hg clone https://bitbucket.org/particon/chpsim
   $ mkvirtualenv -p python3 isaac
   (isaac)$ cd openvpp-agents
   (isaac)$ pip install -r requirements-setup.txt

Now, you should be able to run the test suite:

.. code-block:: console

   (isaac)$ py.test

See :command:`py.test --help` for more information.
