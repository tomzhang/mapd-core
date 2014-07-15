"""
purehuff - A puristic Huffman encoding module for Python

Thomas Grill, 2011
http://grrrr.org/purehuff


Installation:

In the console (terminal application) change to the folder containing this readme.txt file.

To build the package run the following command:
python setup.py build

To install the package (with administrator rights):
sudo python setup.py install
"""

from setuptools import setup

setup(
    name = "purehuff",
    version = "0.2",
    author = "Thomas Grill",
    author_email = "gr@grrrr.org",
    description = ("A puristic Huffman encoding module for Python"),
    license = "GPL",
    keywords = "compression huffman",
    url = "http://grrrr.org",
    packages=['purehuff'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Programming Language :: Python"
    ],
    test_suite="purehuff.__init__"
)
