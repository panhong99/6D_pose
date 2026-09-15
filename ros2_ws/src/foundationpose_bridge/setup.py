from setuptools import setup

setup(name='foundationpose_bridge', version='0.1.0',
      packages=['foundationpose_bridge'],
      data_files=[('share/ament_index/resource_index/packages', ['resource/foundationpose_bridge']),
                  ('share/foundationpose_bridge', ['package.xml'])],
      entry_points={'console_scripts': ['pose_node = foundationpose_bridge.node:main',
                                       'udp_pose_bridge = foundationpose_bridge.udp_pose_bridge:main']})
