import genesis as gs
import numpy as np
import imageio
import os

def run_video_check():
    print("------------------------------------------------")
    print("🐕 CERBERUS VIDEO CHECK")
    print("------------------------------------------------")

    # 1. Init (Headless Mode)
    # We use Vulkan for rendering, but show_viewer=False prevents the window hang
    gs.init(backend=gs.vulkan)

    # 2. Create Scene
    scene = gs.Scene(
        show_viewer=False,
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            world_frame_size=1.0,
            plane_reflection=True,
        ),
        renderer=gs.renderers.Rasterizer(), # Force Rasterizer for speed
    )

    # 3. Add Entities
    plane = scene.add_entity(gs.morphs.Plane())
    
    # Add a Box falling
    cube = scene.add_entity(
        gs.morphs.Box(pos=(0, 0, 1.5), size=(0.4, 0.4, 0.4), fixed=False)
    )

    # 4. Add a Camera (Crucial for video)
    cam = scene.add_camera(
        res=(640, 480),
        pos=(3.0, 3.0, 2.0),
        lookat=(0, 0, 0.5),
        fov=30,
        GUI=False
    )

    scene.build()

    print("[*] Simulating & Recording...")
    frames = []

    # 5. Loop
    for i in range(120): # 2 seconds of video
        scene.step()
        
        # Capture Frame
        # Returns numpy array (H, W, 3)
        rgb = cam.render(rgb=True, depth=False)[0] 
        frames.append(rgb)
        
        if i % 20 == 0:
            print(f"    Frame {i}/120")

    # 6. Save Video
    print("[*] Saving video to 'verify.mp4'...")
    imageio.mimsave('verify.mp4', np.array(frames), fps=60)
    print("✅ SUCCESS: Check 'verify.mp4' in your folder.")

if __name__ == "__main__":
    run_video_check()