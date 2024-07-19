#!/usr/bin/env python3
import os

from setuptools import setup

BASEDIR = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r") as fh:
    long_desc = fh.read()


def get_version():
    """ Find the version of the package"""
    version = None
    version_file = os.path.join(BASEDIR, 'ovos_bm25_solver', 'version.py')
    major, minor, build, alpha = (None, None, None, None)
    with open(version_file) as f:
        for line in f:
            if 'VERSION_MAJOR' in line:
                major = line.split('=')[1].strip()
            elif 'VERSION_MINOR' in line:
                minor = line.split('=')[1].strip()
            elif 'VERSION_BUILD' in line:
                build = line.split('=')[1].strip()
            elif 'VERSION_ALPHA' in line:
                alpha = line.split('=')[1].strip()

            if ((major and minor and build and alpha) or
                    '# END_VERSION_BLOCK' in line):
                break
    version = f"{major}.{minor}.{build}"
    if alpha:
        version += f"a{alpha}"
    return version


def required(requirements_file):
    """ Read requirements file and remove comments and empty lines. """
    with open(os.path.join(BASEDIR, requirements_file), 'r') as f:
        requirements = f.read().splitlines()
        if 'MYCROFT_LOOSE_REQUIREMENTS' in os.environ:
            print('USING LOOSE REQUIREMENTS!')
            requirements = [r.replace('==', '>=').replace('~=', '>=') for r in requirements]
        return [pkg for pkg in requirements
                if pkg.strip() and not pkg.startswith("#")]


PLUGIN_ENTRY_POINTS = [
    'ovos-solver-bm25-plugin=ovos_bm25_solver:BM25CorpusSolver',
    'ovos-solver-bm25qa-plugin=ovos_bm25_solver:BM25QACorpusSolver',
    'ovos-solver-bm25-squad-plugin=ovos_bm25_solver:SquadQASolver',
    'ovos-solver-bm25-freebase-plugin=ovos_bm25_solver:FreebaseQASolver'
]

setup(
    name='ovos-solver-bm25-plugin',
    version=get_version(),
    description='A question solver plugin for OVOS',
    url='https://github.com/TigreGotico/ovos-solver-BM25-plugin',
    author='jarbasai',
    author_email='jarbasai@mailfence.com',
    license='MIT',
    packages=['ovos_bm25_solver'],
    zip_safe=True,
    keywords='OVOS openvoiceos plugin utterance fallback query',
    entry_points={'neon.plugin.solver': PLUGIN_ENTRY_POINTS},
    install_requires=required("requirements.txt"),
    long_description=long_desc,
    long_description_content_type='text/markdown'
)
