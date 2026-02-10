from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'minicar_path_predictor'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='ML-based local path prediction from LiDAR data',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'inference_node = minicar_path_predictor.inference_node:main',
            'ml_nav_node = minicar_path_predictor.ml_nav_node:main',
        ],
    },
)
