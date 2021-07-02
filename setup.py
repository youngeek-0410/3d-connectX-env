import setuptools
import os


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join('.', 'requirements.txt')
    with open(reqs_path, 'r') as f:
        requirements = [line.rstrip() for line in f]
    return requirements


def main():
    with open('README.md', 'r') as fp:
        readme = fp.read()

    setuptools.setup(
        name='3d-connectX-env',
        version='1.0.0',
        description='3D ConnectX for OpenAI Gym.',
        long_description=readme,
        long_description_content_type='text/markdown',
        url='https://github.com/youngeek-0410/3d-connectX-env',
        license='',
        author='Ryusei Ito',
        author_email='31807@toyota.kosen-ac.jp',
        packages=['gym_3d_connectX'],
        install_requires=read_requirements(),
        python_requires='>=3.7, <3.9',
    )


main()
