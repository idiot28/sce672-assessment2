import os
from glob import glob
from setuptools import setup

package_name = 'rm_flatness_nav'

setup(
    name=package_name,
    version='4.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Raian E Rabbani',
    maintainer_email='raian@kfupm.edu.sa',
    description='SCE 672 Assessment 2 (v4): Differential Flatness Pursuit and Motor Voltage Dynamics',
    license='MIT',
    entry_points={
        'console_scripts': [
            'mission_node = rm_flatness_nav.mission_node:main',
            'driver_node = rm_flatness_nav.mission_node:main',
            'planner_node = rm_flatness_nav.mission_node:main',
        ],
    },
)
