#!/usr/bin/env python3
import os

from setuptools import setup

BASEDIR = os.path.abspath(os.path.dirname(__file__))

with open(f"{BASEDIR}/README.md", "r") as fh:
    long_desc = fh.read()


def get_version():
    """ Find the version of the package"""
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
    if int(alpha):
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


SOLVER_ENTRY_POINTS = [
    'ovos-solver-bm25-squad-plugin=ovos_bm25_solver.demo:BM25SquadQASolver',
    'ovos-solver-bm25-freebase-plugin=ovos_bm25_solver.demo:BM25FreebaseQASolver'
]
EVIDENCE_ENTRY_POINT = 'ovos-evidence-solver-bm25=ovos_bm25_solver.base:BM25SolverPlugin'
RERANKER_ENTRY_POINT = 'ovos-choice-solver-bm25=ovos_bm25_solver.base:BM25MultipleChoiceSolver'
EMBED_ENTRY_POINT = 'ovos-json-embeddings-plugin=ovos_bm25_solver.embed:JsonEmbeddingsDB'
TXT_EMBED_ENTRY_POINT = 'ovos-bm25-embeddings-plugin=ovos_bm25_solver.embed:BM25TextEmbeddingsStore'

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
    entry_points={'neon.plugin.solver': SOLVER_ENTRY_POINTS,
                  'opm.embeddings': EMBED_ENTRY_POINT,
                  'opm.embeddings.text': TXT_EMBED_ENTRY_POINT,
                  "opm.solver.reading_comprehension": EVIDENCE_ENTRY_POINT,
                  "opm.solver.multiple_choice": RERANKER_ENTRY_POINT},
    install_requires=required("requirements.txt"),
    long_description=long_desc,
    long_description_content_type='text/markdown'
)
