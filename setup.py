# Copyright (C) 2025, Thibault Hallouin, INRAE.
from setuptools import setup
import json


with open("README.md", "r") as fd:
    long_desc = fd.read()

with open('evalhyd/vigicrues/version.py') as fv:
    exec(fv.read())


def read_requirements(filename):
    requires = []
    with open(filename, 'r') as fr:
        for line in fr:
            package = line.strip()
            if package:
                requires.append(package)

    return requires


def read_authors(filename):
    authors = []
    with open(filename, 'r') as fz:
        meta = json.load(fz)
        for author in meta['creators']:
            name = author['name'].strip()
            authors.append(name)

    return ', '.join(authors)


setup(
    name='evalhyd-vigicrues',

    version=__version__,

    description='A Python add-on to evalhyd providing pre- and post-'
                'processing functionalities specific to VigiCrues forecasts',
    long_description=long_desc,
    long_description_content_type="text/markdown",

    download_url="https://gitlab.irstea.fr/hycar-hydro/evalhyd/evalhyd-vigicrues",
    project_urls={
        'Bug Tracker': 'https://gitlab.irstea.fr/HYCAR-Hydro/evalhyd/evalhyd-vigicrues/-/issues',
        'Source Code': 'https://gitlab.irstea.fr/hycar-hydro/evalhyd/evalhyd-vigicrues',
    },

    author=read_authors('.zenodo.json'),

    license='GPLv3',

    classifiers=[
        'Natural Language :: English',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Hydrology',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python'
    ],

    python_requires=">=3.8",

    packages=[
        'evalhyd.vigicrues',
        'evalhyd.vigicrues.evaluate',
        'evalhyd.vigicrues.plot',
        'evalhyd.vigicrues.read',
    ],

    package_data={
        'evalhyd.vigicrues.evaluate': [
            'evald.toml',
            'evalp.toml'
        ]
    },

    install_requires=read_requirements('requirements.txt')
)
