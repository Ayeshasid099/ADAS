"""
render_3d.py
------------------
STAGE 4 wiring: turns everything your live pipeline ALREADY computes each
frame -- the `vehicles` list (box, name, distance, risk_color) AND the
`drivable_mask` / `lane_mask` from your segmentation models -- into the
Stage 3 renderer's 3D view.

No MiDaS, no extra model calls for placing objects: reuses distances you
already trust (REAL_WIDTHS pinhole math). The road surface is the ACTUAL
shape your drivable-area/lane models detected, not a placeholder grid.

Usage (inside adas_full_pipeline.py's frame loop, after `vehicles`,
`drivable_mask`, and `lane_mask` are built):

    from car_renderer_3d import Scene3D
    from render_3d import render_3d_view

    scene_3d = Scene3D(frame.shape, FOCAL_LENGTH)   # build ONCE, outside the loop
    ...
    canvas_3d = render_3d_view(vehicles, frame.shape, FOCAL_LENGTH, scene_3d,
                                drivable_mask=drivable_mask, lane_mask=lane_mask)
    cv2.imshow("3D View", canvas_3d)
"""

from projection_3d import project_box_ground_point


def render_3d_view(vehicles, frame_shape, focal_length, scene,
                    drivable_mask=None, lane_mask=None,
                    potholes=None, humps=None, signs=None):
    """vehicles: the SAME list of dicts adas_full_pipeline.py already
    builds each frame -- each dict needs 'box', 'name', 'distance', and
    either 'risk_color' or 'class_color' for the object's fill color.
    drivable_mask / lane_mask: boolean masks from seg_utils.build_mask.
    potholes / humps: lists of (x1,y1,x2,y2) boxes.
    signs: list of (name, x1,y1,x2,y2) -- traffic lights/signs.
    All of these are already computed in the main loop -- pass them
    straight through, no extra work needed."""
    canvas = scene.new_canvas()
    scene.draw_ground_grid(canvas)
    scene.draw_road(canvas, drivable_mask=drivable_mask, lane_mask=lane_mask)
    scene.draw_hazards(canvas, potholes=potholes, humps=humps, signs=signs)

    # draw far-to-near so nearer objects correctly overlap farther ones
    for v in sorted(vehicles, key=lambda v: -v["distance"]):
        X, Y, Z = project_box_ground_point(v["box"], v["distance"], frame_shape, focal_length)
        color = v.get("risk_color") or v.get("class_color") or (200, 200, 200)
        # mesh_kind lets callers (e.g. animal detections with dynamic,
        # unpredictable class names like "dog"/"deer"/"cow") pick a
        # specific mesh shape without renaming the object itself.
        mesh_kind = v.get("mesh_kind", v["name"])
        scene.draw_object(canvas, mesh_kind, position=(X, Y, Z), color=color)

    return canvas
