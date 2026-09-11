"""
car_renderer_3d.py
------------------
STAGE 3 of the 3D-engine roadmap: an actual 3D renderer. Draws real 3D
geometry (box meshes with painter's-algorithm face sorting + flat shading)
at the (X, Y, Z) positions Stage 2 computed, onto a dark Tesla-HUD-style
canvas -- genuine 3D geometry pushed through the SAME pinhole camera model
as projection_3d.py, not a flat 2D sprite.

v2 changes:
  - Separate mesh shapes per class (car / truck / bus / person / bike) --
    not every detection drawn as the same car-shaped box anymore.
  - draw_road() paints the ACTUAL drivable-area + lane masks from your
    segmentation models (seg_utils.build_mask output) directly onto the
    canvas. Because this 3D canvas is built at the SAME resolution and
    SAME camera (focal length, frame size) as the original video frame,
    the mask is already in the exact right place -- no reprojection
    needed, no approximation. That's the road "constructed from the
    model", not a generic placeholder grid.

v3 changes:
  - "car" now uses a sedan-shaped mesh (sloped windshield + rear window,
    not a flat-topped box) -- closer to an actual car silhouette.
    Cars stay COLORED by risk (green/orange/red), not grayscale --
    grayscale is only ever the separate Tesla-view 2D window, never this
    3D one.
  - draw_hazards() places pothole / speed-bump / traffic-light markers
    directly on the road, same direct-pixel-placement logic as draw_road
    (same camera, so no reprojection math needed).
  - Road/lane default colors changed to light-blue road + a bright
    yellow lane highlight (see draw_road's defaults).

Why a hand-rolled rasterizer instead of pyrender/OpenGL: pyrender needs a
working OpenGL context (OSMesa/EGL/pyglet), unreliable on a typical
Windows student machine and a heavy dependency chain for what is
fundamentally boxes. This file is pure numpy + OpenCV -- nothing extra to
install.

Usage:
    from car_renderer_3d import Scene3D

    scene = Scene3D(frame_shape=(720, 1280, 3), focal_length=800)
    canvas = scene.new_canvas()
    scene.draw_ground_grid(canvas)
    scene.draw_road(canvas, drivable_mask, lane_mask)   # real road shape
    scene.draw_object(canvas, "car", position=(2.5, 15.0, -0.9), color=(80, 200, 255))
    scene.draw_object(canvas, "person", position=(-1.0, 6.0, -0.9), color=(120, 255, 120))
    cv2.imshow("3D View", canvas)
"""

import cv2
import numpy as np

from projection_3d import unproject_point
from seg_utils import blend_mask


def _box_mesh(length, width, height):
    """8-vertex box, ground-contact at z=0. Local frame: x=lateral
    (width), y=forward (length, +y = front), z=height."""
    hl, hw = length / 2.0, width / 2.0
    verts = np.array([
        [-hw, -hl, 0], [hw, -hl, 0], [hw, hl, 0], [-hw, hl, 0],
        [-hw, -hl, height], [hw, -hl, height], [hw, hl, height], [-hw, hl, height],
    ], dtype=np.float64)
    faces = [
        [0, 1, 2, 3], [4, 5, 6, 7],
        [0, 1, 5, 4], [2, 3, 7, 6],
        [1, 2, 6, 5], [3, 0, 4, 7],
    ]
    return verts, faces


def _car_like_mesh(length, width, height, cabin_ratio=0.55):
    """Body box + smaller cabin box on top -- recognizable car/truck/bus
    silhouette (hood + cabin) instead of a plain brick."""
    body, body_faces = _box_mesh(length, width, height)
    hl, hw = length / 2.0, width / 2.0

    cab_h = height * cabin_ratio
    cab_hl = hl * 0.55
    cab_y0 = -hl * 0.05
    cabin = np.array([
        [-hw * 0.95, cab_y0 - cab_hl, height], [hw * 0.95, cab_y0 - cab_hl, height],
        [hw * 0.95, cab_y0 + cab_hl, height], [-hw * 0.95, cab_y0 + cab_hl, height],
        [-hw * 0.85, cab_y0 - cab_hl, height + cab_h], [hw * 0.85, cab_y0 - cab_hl, height + cab_h],
        [hw * 0.85, cab_y0 + cab_hl, height + cab_h], [-hw * 0.85, cab_y0 + cab_hl, height + cab_h],
    ], dtype=np.float64)
    cabin_faces = [[4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [3, 0, 4, 7]]

    vertices = np.vstack([body, cabin])
    faces = body_faces + [[i + 8 for i in f] for f in cabin_faces]
    return vertices, faces


def _sedan_mesh(length=4.3, width=1.8, height=1.15, cabin_height=0.55):
    """Body box (hood/trunk deck) + a HEXAGONAL cabin (sloped windshield,
    flat roof, sloped rear window, 2 side windows) instead of a flat-
    topped box cabin -- reads as an actual sedan silhouette, closer to
    the reference Tesla-visualizer look, instead of a brick with a
    smaller brick on top."""
    body, body_faces = _box_mesh(length, width, height)
    hl, hw = length / 2.0, width / 2.0
    cab_hw = hw * 0.85

    cab_hl = hl * 0.62          # cabin footprint half-length
    cab_center_y = -hl * 0.05   # slightly toward the rear, like real sedans
    y_fb = cab_center_y + cab_hl          # windshield base (forward-most)
    y_ft = cab_center_y + cab_hl * 0.45   # roof front (windshield rake)
    y_rt = cab_center_y - cab_hl * 0.55   # roof rear
    y_rb = cab_center_y - cab_hl          # rear-window base (rearmost)

    z_lo = height                 # sits on the body's flat deck
    z_hi = height + cabin_height  # roof height

    # 8 verts: 4 profile points (fb, ft, rt, rb) x 2 sides (L=-x, R=+x)
    cabin = np.array([
        [-cab_hw, y_fb, z_lo], [cab_hw, y_fb, z_lo],   # 0 fb_L, 1 fb_R
        [cab_hw, y_ft, z_hi], [-cab_hw, y_ft, z_hi],   # 2 ft_R, 3 ft_L
        [-cab_hw, y_rt, z_hi], [cab_hw, y_rt, z_hi],   # 4 rt_L, 5 rt_R
        [cab_hw, y_rb, z_lo], [-cab_hw, y_rb, z_lo],   # 6 rb_R, 7 rb_L
    ], dtype=np.float64)

    cabin_faces = [
        [0, 1, 2, 3],  # windshield (slopes up from hood to roof-front)
        [3, 2, 5, 4],  # roof (flat)
        [4, 5, 6, 7],  # rear window (slopes down from roof-rear to trunk)
        [0, 3, 4, 7],  # left side window
        [1, 6, 5, 2],  # right side window
    ]

    vertices = np.vstack([body, cabin])
    faces = body_faces + [[i + 8 for i in f] for f in cabin_faces]
    return vertices, faces



def _person_mesh(height=1.7, width=0.45, depth=0.35):
    """Two stacked boxes -- narrow 'legs/torso' block + a slightly smaller
    'head' block on top -- reads as a standing person, not a car brick."""
    torso, torso_faces = _box_mesh(depth, width, height * 0.82)
    hw, hd = width * 0.35, depth * 0.35
    head = np.array([
        [-hw, -hd, height * 0.82], [hw, -hd, height * 0.82],
        [hw, hd, height * 0.82], [-hw, hd, height * 0.82],
        [-hw, -hd, height], [hw, -hd, height], [hw, hd, height], [-hw, hd, height],
    ], dtype=np.float64)
    head_faces = [[4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [3, 0, 4, 7]]

    vertices = np.vstack([torso, head])
    faces = torso_faces + [[i + 8 for i in f] for f in head_faces]
    return vertices, faces


def _animal_mesh(length=0.9, width=0.35, height=0.6, head_ratio=0.35):
    """Low quadruped-style silhouette: horizontal body box + a smaller
    head/neck box forward and above -- reads as an animal (dog/deer/cow-
    scale, lightweight mesh, no extra depth model needed) rather than a
    vehicle brick. Works reasonably for any generic 4-legged animal class
    since the exact species isn't known ahead of time (model.names is
    read dynamically -- see adas_core.py)."""
    body, body_faces = _box_mesh(length, width, height * 0.7)
    hl = length / 2.0
    head_w = width * head_ratio + 0.1
    head_h = height * 0.4
    head_y0 = hl * 0.75
    head = np.array([
        [-head_w / 2, head_y0 - head_w / 2, height * 0.7],
        [head_w / 2, head_y0 - head_w / 2, height * 0.7],
        [head_w / 2, head_y0 + head_w / 2, height * 0.7],
        [-head_w / 2, head_y0 + head_w / 2, height * 0.7],
        [-head_w / 2, head_y0 - head_w / 2, height * 0.7 + head_h],
        [head_w / 2, head_y0 - head_w / 2, height * 0.7 + head_h],
        [head_w / 2, head_y0 + head_w / 2, height * 0.7 + head_h],
        [-head_w / 2, head_y0 + head_w / 2, height * 0.7 + head_h],
    ], dtype=np.float64)
    head_faces = [[4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [3, 0, 4, 7]]

    vertices = np.vstack([body, head])
    faces = body_faces + [[i + 8 for i in f] for f in head_faces]
    return vertices, faces


def _bike_mesh(length=1.9, width=0.5, height=1.55, rider_ratio=0.6):
    """Low, slim frame box + a rider-torso box above the rear half --
    reads as a two-wheeler + rider silhouette, distinct from a car."""
    frame, frame_faces = _box_mesh(length, width, height * 0.35)
    hl = length / 2.0
    rider_h = height * rider_ratio
    rw, rd = width * 0.9, length * 0.22
    ry0 = hl * 0.15
    rider = np.array([
        [-rw / 2, ry0 - rd, height * 0.35], [rw / 2, ry0 - rd, height * 0.35],
        [rw / 2, ry0 + rd, height * 0.35], [-rw / 2, ry0 + rd, height * 0.35],
        [-rw / 2, ry0 - rd, height * 0.35 + rider_h], [rw / 2, ry0 - rd, height * 0.35 + rider_h],
        [rw / 2, ry0 + rd, height * 0.35 + rider_h], [-rw / 2, ry0 + rd, height * 0.35 + rider_h],
    ], dtype=np.float64)
    rider_faces = [[4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [3, 0, 4, 7]]

    vertices = np.vstack([frame, rider])
    faces = frame_faces + [[i + 8 for i in f] for f in rider_faces]
    return vertices, faces


# One mesh per class your model actually outputs (adas_full_pipeline.py's
# REAL_WIDTHS keys): car, bus, truck, person, rider, bike, motor.
# Unknown class names fall back to the plain car mesh.
_MESH_BUILDERS = {
    "car":    lambda: _sedan_mesh(4.3, 1.8, 1.15),
    "truck":  lambda: _car_like_mesh(7.0, 2.3, 2.6, cabin_ratio=0.4),
    "bus":    lambda: _car_like_mesh(9.5, 2.5, 3.0, cabin_ratio=0.85),
    "person": lambda: _person_mesh(),
    "rider":  lambda: _bike_mesh(),
    "bike":   lambda: _bike_mesh(length=1.7, width=0.45, height=1.5),
    "motor":  lambda: _bike_mesh(length=2.0, width=0.55, height=1.6),
    "animal": lambda: _animal_mesh(),
}


class Scene3D:
    def __init__(self, frame_shape, focal_length, bg_color=(15, 15, 15)):
        self.frame_shape = frame_shape
        self.focal_length = focal_length
        self.bg_color = bg_color
        # build every mesh once up front -- geometry is static, only the
        # per-frame position/color/yaw changes
        self._mesh_library = {name: build() for name, build in _MESH_BUILDERS.items()}
        self._default_mesh = self._mesh_library["car"]

    def new_canvas(self):
        h, w = self.frame_shape[:2]
        canvas = np.empty((h, w, 3), dtype=np.uint8)
        canvas[:] = self.bg_color
        return canvas

    def _project(self, world_xyz):
        X, Y, Z = world_xyz
        if Y <= 0.3:
            return None  # behind/on top of the camera -- don't draw
        u, v = unproject_point(X, Y, Z, self.frame_shape, self.focal_length)
        return (u, v), Y

    def draw_ground_grid(self, canvas, extent=40, step=4, ground_z=-1.6, color=(50, 50, 58)):
        """Faint perspective floor grid, drawn BEHIND draw_road() -- gives
        spatial reference beyond the actual detected road edge."""
        for x in range(-extent, extent + 1, step):
            p1 = self._project((x, 1.0, ground_z))
            p2 = self._project((x, extent, ground_z))
            if p1 and p2:
                cv2.line(canvas, tuple(np.int32(p1[0])), tuple(np.int32(p2[0])), color, 1, cv2.LINE_AA)
        for y in range(4, extent + 1, step):
            p1 = self._project((-extent, y, ground_z))
            p2 = self._project((extent, y, ground_z))
            if p1 and p2:
                cv2.line(canvas, tuple(np.int32(p1[0])), tuple(np.int32(p2[0])), color, 1, cv2.LINE_AA)

    def draw_road(self, canvas, drivable_mask=None, lane_mask=None,
                   drivable_color=(235, 206, 135), lane_color=(0, 225, 255),
                   drivable_alpha=0.55, lane_alpha=0.9):
        """Paint the REAL road/lane shape your segmentation models found,
        straight onto this canvas. Exact placement -- this canvas shares
        the source frame's resolution and camera params, so the mask
        needs no reprojection, just a direct blend. Call this AFTER
        draw_ground_grid (road covers the grid where it exists) and
        BEFORE draw_object (vehicles/people draw on top of the road).
        Defaults: light-blue road surface, bright yellow lane highlight."""
        if drivable_mask is not None and drivable_mask.any():
            canvas[:] = blend_mask(canvas, drivable_mask, color=drivable_color, alpha=drivable_alpha)
        if lane_mask is not None and lane_mask.any():
            canvas[:] = blend_mask(canvas, lane_mask, color=lane_color, alpha=lane_alpha)

    def draw_hazards(self, canvas, potholes=None, humps=None, signs=None,
                      pothole_color=(230, 0, 255), hump_color=(0, 145, 255),
                      sign_color=(0, 215, 255)):
        """Places pothole / speed-bump / traffic-light-or-sign markers
        directly on the road. Same reasoning as draw_road: this canvas is
        the SAME camera/resolution as the source frame, so a detected
        box's pixel coordinates are already exactly where it belongs here
        too -- no distance/3D math needed for flat road-surface hazards.

        potholes, humps : lists of (x1, y1, x2, y2) boxes, e.g. straight
                           from adas_full_pipeline.py's `potholes`/`humps`.
        signs            : list of (name, x1, y1, x2, y2) tuples, e.g.
                           straight from adas_full_pipeline.py's `signs`
                           (covers both 'traffic light' and 'traffic sign').
        Distinct colors so they don't get confused with each other or
        with vehicles: pothole = magenta, hump = orange, sign/light = amber.
        """
        for (x1, y1, x2, y2) in (potholes or []):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            axes = (max(4, (x2 - x1) // 2), max(3, (y2 - y1) // 4))
            cv2.ellipse(canvas, (cx, y2), axes, 0, 0, 360, pothole_color, -1, cv2.LINE_AA)
            cv2.ellipse(canvas, (cx, y2), axes, 0, 0, 360, (0, 0, 0), 1, cv2.LINE_AA)

        for (x1, y1, x2, y2) in (humps or []):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            pts = np.array([[x1, y2], [x2, y2], [(x1 + x2) // 2, y1]], dtype=np.int32)
            cv2.fillConvexPoly(canvas, pts, hump_color, cv2.LINE_AA)
            cv2.polylines(canvas, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)

        for (name, x1, y1, x2, y2) in (signs or []):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), sign_color, 2, cv2.LINE_AA)
            cv2.putText(canvas, name.upper(), (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, sign_color, 1, cv2.LINE_AA)

    def draw_object(self, canvas, kind, position, color=(80, 200, 255), yaw=0.0, scale=1.0):
        """kind: class name string ('car', 'truck', 'bus', 'person',
        'rider', 'bike', 'motor') -- picks the matching mesh shape.
        Unrecognized names fall back to the car mesh.
        position: (X, Y, Z) meters -- ground-contact point.
        yaw: rotation around the vertical axis, radians (0 = facing away
        from the camera -- oncoming traffic would use yaw=pi)."""
        vertices, faces = self._mesh_library.get(kind, self._default_mesh)

        X0, Y0, Z0 = position
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        verts = vertices * scale
        rx = verts[:, 0] * cos_y - verts[:, 1] * sin_y
        ry = verts[:, 0] * sin_y + verts[:, 1] * cos_y
        world = np.stack([rx + X0, ry + Y0, verts[:, 2] + Z0], axis=1)

        projected, depths = [], []
        for wv in world:
            res = self._project(tuple(wv))
            if res is None:
                return  # any vertex behind camera -> skip whole object (simple, avoids clip math)
            (u, v), depth = res
            projected.append((u, v))
            depths.append(depth)
        projected = np.array(projected)
        depths = np.array(depths)

        face_order = sorted(range(len(faces)), key=lambda i: -np.mean(depths[faces[i]]))

        for i in face_order:
            face = faces[i]
            pts = np.int32(projected[face])

            verts3d = world[face]
            normal = np.cross(verts3d[1] - verts3d[0], verts3d[2] - verts3d[0])
            norm_len = np.linalg.norm(normal)
            shade = 0.55 + 0.45 * max(0.0, normal[2] / norm_len) if norm_len > 1e-6 else 1.0
            shaded_color = tuple(int(c * shade) for c in color)

            cv2.fillConvexPoly(canvas, pts, shaded_color, cv2.LINE_AA)
            cv2.polylines(canvas, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)

    def draw_car(self, canvas, position, color=(80, 200, 255), yaw=0.0, scale=1.0):
        """Back-compat alias for older calls -- same as draw_object(kind='car', ...)."""
        self.draw_object(canvas, "car", position, color, yaw, scale)
