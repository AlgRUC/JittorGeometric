from setuptools import setup, find_packages, Extension


setup(
    name='jittor_geometric',
    version='0.1',
    # author='Your Name',有一些信息不清楚发行版该怎么填，就先注释掉了
    # author_email='your.email@example.com',恳请大佬们写一下吧
    packages=find_packages(),
    package_data={
        'jittor_geometric': ['ops/cpp/*.cc', 'ops/cpp/*.h'],
    },
    # description='A brief description of the library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    # install_requires=[
    #     "jittor==1.3.8.5",
    #     "numpy==2.1.3",
    #     "pandas==2.2.3",
    #     "pyparsing==3.2.0",
    #     "scikit-learn==1.5.2",
    #     "scipy==1.14.1",
    #     "six==1.16.0",
    #     "tqdm==4.66.4",
    # ],
    classifiers=[
    #
    ],
    # python_requires='',
)
