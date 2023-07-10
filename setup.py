from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='signxai',
    version='0.0.1',
    packages=['signxai'],
    url='https://github.com/nilsgumpfer/SIGN-XAI',
    license='BSD 2-Clause License',
    author='Nils Gumpfer',
    author_email='nils.gumpfer@kite.thm.de',
    maintainer='Nils Gumpfer',
    maintainer_email='nils.gumpfer@kite.thm.de',
    description='SIGNed explanations: Unveiling relevant features by reducing bias',
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['XAI', 'SIGN', 'LRP'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: BSD License',
        'Development Status :: 4 - Beta',
        # 'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=['tensorflow>=2.2.0', 'matplotlib>=3.3.4'],
    include_package_data=True,
)
