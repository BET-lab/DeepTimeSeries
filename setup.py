from setuptools import setup, find_packages
from git import Repo


# git commit sha is used as version until the first stable release.
# Only the first 7 characters are taken.
commit = Repo(search_parent_directories=True).head.object.hexsha[:7]
version = commit

# Get dependencies from requirements.txt file.
with open('requirements.txt', 'r') as f:
    install_requires = f.readlines()

setup(
    name='deep-time-series',
    version=version,
    description='Deep learning library for '
        'time series forecasting based on PyTorch.',
    install_requires=install_requires,
    author='Sangwon Lee',
    author_email='lsw91.main@gmail.com',
    # packages=[
    #     'deep_time_series',
    # ],
    packages=find_packages(),
    # package_data={
    #     '???': ['???/**/*'],
    # },
    python_requires='>=3.8',
    zip_safe=False
)