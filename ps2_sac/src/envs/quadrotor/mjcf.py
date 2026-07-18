QUADROTOR_MJCF = """
<mujoco model="quadrotor_playback">
  <option timestep="0.02" gravity="0 0 -9.81"/>

  <worldbody>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"/>

    <!-- Ceiling plane at z = 3.0 -->
    <geom name="ceiling" type="plane" pos="0 0 3" size="10 10 0.01"
          rgba="1 0 0 0.15"/>

    <body name="quadrotor" pos="0 0 2">
      <freejoint name="root"/>

      <!-- central body -->
      <geom type="box" size="0.08 0.08 0.03" rgba="0.1 0.1 0.1 1"/>

      <!-- arms -->
      <geom type="box" pos="0.18 0 0" size="0.18 0.015 0.015" rgba="0.2 0.2 0.8 1"/>
      <geom type="box" pos="-0.18 0 0" size="0.18 0.015 0.015" rgba="0.2 0.2 0.8 1"/>
      <geom type="box" pos="0 0.18 0" size="0.015 0.18 0.015" rgba="0.2 0.8 0.2 1"/>
      <geom type="box" pos="0 -0.18 0" size="0.015 0.18 0.015" rgba="0.2 0.8 0.2 1"/>

      <!-- rotors -->
      <geom type="cylinder" pos="0.35 0 0" size="0.07 0.005" rgba="0 0 0 1"/>
      <geom type="cylinder" pos="-0.35 0 0" size="0.07 0.005" rgba="0 0 0 1"/>
      <geom type="cylinder" pos="0 0.35 0" size="0.07 0.005" rgba="0 0 0 1"/>
      <geom type="cylinder" pos="0 -0.35 0" size="0.07 0.005" rgba="0 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""