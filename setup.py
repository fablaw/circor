from setuptools import setup
from setuptools import find_packages


setup(name='circor',
      description="package description",
      packages=["circor"]) # You can have several packages, try it


# list dependencies from file
with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content]

setup(name='circor',
      description="package description",
      packages=find_packages(), # NEW: find packages automatically
      install_requires=requirements) # NEW
