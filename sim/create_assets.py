import os
from pathlib import Path

def create_assets():
    path = Path("assets/mini_pupper")
    path.mkdir(parents=True, exist_ok=True)
    
    # Official Mini Pupper v2 Physical Constants
    urdf_content = """<?xml version="1.0"?>
<robot name="mini_pupper">
  <link name="base_link">
    <visual>
      <geometry><box size="0.209 0.108 0.045"/></geometry>
      <material name="yellow"><color rgba="1 0.8 0 1"/></material>
    </visual>
    <collision>
      <geometry><box size="0.209 0.108 0.045"/></geometry>
    </collision>
    <inertial>
      <mass value="0.506"/>
      <inertia ixx="0.0006" ixy="0.0" ixz="0.0" iyy="0.0015" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>
"""
    legs = [
        ("fl", 0.09, 0.05), ("fr", 0.09, -0.05),
        ("bl", -0.09, 0.05), ("br", -0.09, -0.05)
    ]

    for name, x, y in legs:
        urdf_content += f"""
  <joint name="{name}_hip_joint" type="revolute">
    <parent link="base_link"/><child link="{name}_hip"/>
    <origin xyz="{x} {y} 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.8" upper="0.8" effort="3.0" velocity="6.0"/>
  </joint>
  <link name="{name}_hip">
    <inertial>
      <mass value="0.06"/>
      <inertia ixx="0.00001" ixy="0.0" ixz="0.0" iyy="0.00001" iyz="0.0" izz="0.00001"/>
    </inertial>
  </link>

  <joint name="{name}_thigh_joint" type="revolute">
    <parent link="{name}_hip"/><child link="{name}_thigh"/>
    <origin xyz="0 {0.02 if y > 0 else -0.02} 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="4.5" effort="3.0" velocity="6.0"/>
  </joint>
  <link name="{name}_thigh">
    <inertial>
      <mass value="0.04"/>
      <inertia ixx="0.00001" ixy="0.0" ixz="0.0" iyy="0.00001" iyz="0.0" izz="0.00001"/>
    </inertial>
  </link>

  <joint name="{name}_calf_joint" type="revolute">
    <parent link="{name}_thigh"/><child link="{name}_calf"/>
    <origin xyz="0 0 -0.1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="-0.5" effort="3.0" velocity="6.0"/>
  </joint>
  <link name="{name}_calf">
    <inertial>
      <mass value="0.02"/>
      <inertia ixx="0.00001" ixy="0.0" ixz="0.0" iyy="0.00001" iyz="0.0" izz="0.00001"/>
    </inertial>
  </link>
"""
    urdf_content += "</robot>"
    
    with open(path / "mini_pupper.urdf", "w") as f:
        f.write(urdf_content)
    print(f"✅ Created structural-correct Mini Pupper URDF.")

if __name__ == "__main__":
    create_assets()