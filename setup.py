from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='signxai',
    version='1.0.0',
    packages=['signxai.methods', 'signxai.methods.innvestigate', 'signxai.methods.innvestigate.tests', 'signxai.methods.innvestigate.tests.tools',
              'signxai.methods.innvestigate.tests.utils', 'signxai.methods.innvestigate.tests.utils.keras',
              'signxai.methods.innvestigate.tests.utils.tests', 'signxai.methods.innvestigate.tests.analyzer',
              'signxai.methods.innvestigate.tools', 'signxai.methods.innvestigate.utils', 'signxai.methods.innvestigate.utils.keras',
              'signxai.methods.innvestigate.utils.tests', 'signxai.methods.innvestigate.utils.tests.cases',
              'signxai.methods.innvestigate.backend', 'signxai.methods.innvestigate.analyzer',
              'signxai.methods.innvestigate.analyzer.canonization', 'signxai.methods.innvestigate.analyzer.relevance_based',
              'signxai.methods.innvestigate.applications', 'signxai.examples', 'signxai.utils'],
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
    install_requires=['tensorflow>=2.2.0', 'matplotlib>=3.3.4', 'requests>=2.27.1'],
    include_package_data=True,
)
