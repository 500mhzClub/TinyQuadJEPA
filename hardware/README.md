# Hardware Build Guide (The Hacker Class)

We are building a research-grade quadruped for about 700. This is significantly cheaper than a Pupper or Go2.

## Bill of Materials
* **Brain:** NVIDIA Jetson Orin Nano (8GB Module) + Tiny Carrier Board.
* **Servos:** 12x **STS3215** Serial Bus Servos.
   * *Crucial:* Do not buy PWM servos. We need the position/load feedback for the JEPA.
* **Camera:** Raspberry Pi Cam v3 (Wide Angle).
* **Power:** 3S LiPo (2200mAh) + External 20A BEC.
   * *Warning:* Do not power servos through the carrier board. You will fry it. Use the BEC.

## Wiring Strategy
* **Split Power:** Battery -> XT60 Splitter.
   * Path A: BEC -> Servo Bus (7.4V).
   * Path B: BEC -> Jetson (5V/12V depending on carrier).
* **Comms:** The Jetson talks to the servos via a UART-to-TTL board (usually /dev/ttyTHS1).

## Assembly Tips
1. **Heat Inserts:** 3D printed threads are not suitable. Use brass M3 heat-set inserts for everything.
2. **Zeroing:** Before you screw on the legs, plug every servo into the PC and set it to ID 1-12 and Position 2048 (Midpoint). If you screw the legs on while the servo is at 0, you will break the legs instantly on boot.
3. **The "Femur" Problem:** Print 4 extra lower leg bones in PETG. They are the mechanical fuse. When the robot falls, these should break before the servo gears do.