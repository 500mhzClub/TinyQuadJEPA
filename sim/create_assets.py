#!/usr/bin/env python3
from pathlib import Path
import math

def box_inertia(m, x, y, z):
    # inertia of a solid box about its center
    Ixx = m * (y*y + z*z) / 12.0
    Iyy = m * (x*x + z*z) / 12.0
    Izz = m * (x*x + y*y) / 12.0
    return Ixx, Iyy, Izz

def write_urdf():
    asset_dir = Path("assets/mini_pupper")
    mesh_dir  = "meshes"
    out_file  = asset_dir / "mini_pupper.urdf"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Base mesh (from your STL bounds it’s already ~meters scale; no 1.82m offset needed)
    base_x, base_y, base_z = 0.1811, 0.0700, 0.0700
    base_m = 0.56
    base_Ixx, base_Iyy, base_Izz = box_inertia(base_m, base_x, base_y, base_z)

    # Leg config: name, base->hip joint origin x,y, and side sign (for your y offsets)
    legs = [
        ("lf",  0.060,  0.035,  1),
        ("lh", -0.060,  0.035,  1),
        ("rf",  0.060, -0.035, -1),
        ("rh", -0.060, -0.035, -1),
    ]

    # Reasonable-ish actuator limits for standing (tune later)
    EFFORT = 12.0
    VEL    = 10.0

    urdf = f"""<?xml version="1.0"?>
<robot name="mini_pupper">
  <material name="yellow"><color rgba="1 0.8 0 1"/></material>
  <material name="black"><color rgba="0.1 0.1 0.1 1"/></material>

  <!-- ================= Base ================= -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_dir}/base.stl"/></geometry>
      <material name="yellow"/>
    </visual>

    <!-- Align collision box to the base mesh which sits roughly on z>=0 -->
    <collision>
      <origin xyz="0 0 {base_z/2:.5f}" rpy="0 0 0"/>
      <geometry><box size="{base_x:.5f} {base_y:.5f} {base_z:.5f}"/></geometry>
    </collision>

    <!-- Put COM roughly mid-height -->
    <inertial>
      <origin xyz="0 0 {base_z/2:.5f}" rpy="0 0 0"/>
      <mass value="{base_m:.5f}"/>
      <inertia ixx="{base_Ixx:.8f}" ixy="0" ixz="0" iyy="{base_Iyy:.8f}" iyz="0" izz="{base_Izz:.8f}"/>
    </inertial>
  </link>
"""

    for name, x, y, side in legs:
        # Hip link
        urdf += f"""
  <!-- ================= {name.upper()} Leg ================= -->
  <joint name="{name}_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="{name}_hip"/>
    <origin xyz="{x:.5f} {y:.5f} 0.01710" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.8" upper="0.8" effort="{EFFORT}" velocity="{VEL}"/>
  </joint>

  <link name="{name}_hip">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_dir}/{name}_hip.stl"/></geometry>
      <material name="black"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="1.57079632679 0 0"/>
      <geometry><cylinder radius="0.015" length="0.04"/></geometry>
    </collision>
    <inertial>
      <mass value="0.08"/>
      <inertia ixx="0.00010" ixy="0" ixz="0" iyy="0.00010" iyz="0" izz="0.00010"/>
    </inertial>
  </link>

  <!-- Thigh -->
  <joint name="{name}_thigh_joint" type="revolute">
    <parent link="{name}_hip"/>
    <child link="{name}_thigh"/>
    <origin xyz="0 {side * 0.01970:.5f} 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="1.5" effort="{EFFORT}" velocity="{VEL}"/>
  </joint>

  <link name="{name}_thigh">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_dir}/{name}_upper_leg.stl"/></geometry>
      <material name="yellow"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.025" rpy="0 0 0"/>
      <geometry><box size="0.02 0.01 0.05"/></geometry>
    </collision>
    <inertial>
      <mass value="0.08"/>
      <inertia ixx="0.00010" ixy="0" ixz="0" iyy="0.00010" iyz="0" izz="0.00010"/>
    </inertial>
  </link>

  <!-- Calf: IMPORTANT: allow 0 so qpos0 doesn't violate limits -->
  <joint name="{name}_calf_joint" type="revolute">
    <parent link="{name}_thigh"/>
    <child link="{name}_calf"/>
    <origin xyz="0 {side * 0.00475:.5f} -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="0.5" effort="{EFFORT}" velocity="{VEL}"/>
  </joint>

  <link name="{name}_calf">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_dir}/{name}_lower_leg.stl"/></geometry>
      <material name="black"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.03" rpy="0 0 0"/>
      <geometry><box size="0.015 0.01 0.06"/></geometry>
    </collision>
    <inertial>
      <mass value="0.04"/>
      <inertia ixx="0.00002" ixy="0" ixz="0" iyy="0.00002" iyz="0" izz="0.00002"/>
    </inertial>
  </link>

  <!-- Foot -->
  <joint name="{name}_foot_fixed" type="fixed">
    <parent link="{name}_calf"/>
    <child link="{name}_foot"/>
    <origin xyz="0 0 -0.056" rpy="0 0 0"/>
  </joint>

  <link name="{name}_foot">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{mesh_dir}/foot.stl"/></geometry>
      <material name="black"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><sphere radius="0.012"/></geometry>
    </collision>
    <inertial>
      <mass value="0.01"/>
      <inertia ixx="0.000001" ixy="0" ixz="0" iyy="0.000001" iyz="0" izz="0.000001"/>
    </inertial>
  </link>
"""
    urdf += "\n</robot>\n"
    out_file.write_text(urdf)
    print(f"✅ Wrote: {out_file}")

if __name__ == "__main__":
    write_urdf()
