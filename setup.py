from setuptools import setup, find_packages, Extension


setup(
    name='jittor_geometric',
    version='0.1',
    # author='Your Name',
    # author_email='your.email@example.com',
    packages=find_packages(),
    package_data={
        'jittor_geometric': ['ops/cpp/*.cc', 'ops/cpp/*.h'],
    },
    # description='A brief description of the library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
    #
    ],
    # python_requires='',
)
