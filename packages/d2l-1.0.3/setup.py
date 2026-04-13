from setuptools import setup, find_packages
import d2l

requirements = [
    'jupyter==1.1.1',
    'numpy>=2.1.0',
    'matplotlib==3.7.2',
    'matplotlib-inline==0.1.6',
    'requests>=2.32.2',
    'pandas==2.3.3',
    'scipy==1.17.1'
]

setup(
    name='d2l',
    version=d2l.__version__,
    python_requires='>=3.8',
    author='D2L Developers',
    author_email='d2l.devs@gmail.com',
    url='https://d2l.ai',
    description='Dive into Deep Learning, patched with https://github.com/d2l-ai/d2l-en/issues/2625',
    license='MIT-0',
    packages=find_packages(),
    zip_safe=True,
    install_requires=requirements,
)
