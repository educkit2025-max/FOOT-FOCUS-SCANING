"""FOCUS-SfM: coloured multi-view foot surface from video frames.

Follows FOCUS (github.com/OllieBoyne/FOCUS, MIT licence, Boyne & Cipolla 3DV 2025): per-frame TOC
prediction, TOC nearest-neighbour matching with sub-pixel refinement, COLMAP bundle adjustment on those
matches, triangulation, TOC-guided cropping and screened Poisson meshing with per-point colour.

The SfM result has no metric scale; `to_foot_frame` scales it with the A4-calibrated foot length.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

FRAME_W, FRAME_H = 480, 640
NUM_VIEWS = 20
MIN_MASK_COVERAGE = 0.02
COLMAP_CORRESPONDENCES = 4000
FUSION_CORRESPONDENCES = 20000
TOC_MATCH_MAX_DIST = 0.002
SUBPIXEL_STEPS = 8
REPROJ_MAX_PX = 3.0
# FIND template bounds (metres): x heel→toe, y across the foot, z up.
TEMPLATE_MIN = np.array([-0.124274, -0.046119, -0.053607])
TEMPLATE_MAX = np.array([0.139912, 0.050383, 0.096478])
TOC_Z_KEEP = (-0.02, 0.8)
POISSON = {"depth": 8, "samplespernode": 1.5, "iters": 8, "cgdepth": 0, "pointweight": 0}
TARGET_FACES = 9000
PAPER_WHITE = 0.92
COLOUR_EDGE_MIN_PX = 5.0

_MODEL = None


@dataclass
class View:
    key: str
    rgb: np.ndarray  # H x W x 3, 0..1
    toc: np.ndarray  # H x W x 3, 0..1
    normal: np.ndarray  # H x W x 3, network camera frame
    mask: np.ndarray  # H x W bool
    colour: np.ndarray | None = None  # H x W x 4: white-balanced rgb + distance to the mask edge (px)
    idx: int = 0
    _tree: cKDTree | None = None
    _pix: np.ndarray | None = None

    def tree(self) -> tuple[cKDTree, np.ndarray]:
        if self._tree is None:
            ys, xs = np.nonzero(self.mask)
            self._pix = np.stack([xs, ys], axis=1)
            self._tree = cKDTree(self.toc[ys, xs])
        return self._tree, self._pix


class Correspondences:
    """Samples of one TOC value observed in several views (FOCUS `Correspondence`, as flat arrays)."""

    def __init__(self) -> None:
        self.toc = np.zeros((0, 3))
        self.c: list[np.ndarray] = []
        self.v: list[np.ndarray] = []
        self.xy: list[np.ndarray] = []
        self.rgb: list[np.ndarray] = []
        self.n: list[np.ndarray] = []

    def add(self, c, v, xy, rgb, n) -> None:
        self.c.append(np.asarray(c, np.int64))
        self.v.append(np.asarray(v, np.int64))
        self.xy.append(np.asarray(xy, np.float64))
        self.rgb.append(np.asarray(rgb, np.float64))
        self.n.append(np.asarray(n, np.float64))

    def arrays(self) -> tuple[np.ndarray, ...]:
        return tuple(np.concatenate(a) for a in (self.c, self.v, self.xy, self.rgb, self.n))


def letterbox(img_rgb: np.ndarray) -> np.ndarray:
    """FOCUS `resize_preserve_aspect` to 480 x 640 with zero padding."""
    h, w = img_rgb.shape[:2]
    if h > w:
        nh, nw = FRAME_H, int(w * FRAME_H / h)
    else:
        nh, nw = int(h * FRAME_W / w), FRAME_W
    out = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    py, px = (FRAME_H - nh) // 2, (FRAME_W - nw) // 2
    out[py : py + nh, px : px + nw] = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def _largest_island(mask: np.ndarray) -> np.ndarray:
    n, labels = cv2.connectedComponents(mask.astype(np.uint8))
    if n <= 1:
        return mask
    counts = np.bincount(labels.ravel())
    return labels == (np.argmax(counts[1:]) + 1)


def _paper_gain(img: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Per-channel gain that makes the white A4 sheet (bright, unsaturated, off-foot) read as white."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    off = (~cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)) & (img.max(axis=2) > 8)
    if off.sum() < 2000:
        return None
    v = hsv[..., 2].astype(np.float64)
    paper = off & (hsv[..., 1] < 70) & (v >= np.percentile(v[off], 80))
    if paper.sum() < 500:
        return None
    ref = np.median(img[paper].astype(np.float64), axis=0) / 255.0
    return np.clip(PAPER_WHITE / np.maximum(ref, 1e-3), 0.8, 3.0)


def _colour_layer(img: np.ndarray, mask: np.ndarray, gain: np.ndarray) -> np.ndarray:
    rgb = np.clip(img.astype(np.float64) / 255.0 * gain, 0, 1)
    edge = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3).astype(np.float64)
    return np.dstack([rgb, edge])


def _model(model_path: Path):
    global _MODEL
    if _MODEL is None:
        import torch

        from app.agents.focus.toc_model import FootPredictorModel

        torch.set_num_threads(max(1, torch.get_num_threads()))
        _MODEL = FootPredictorModel.load(model_path)
    return _MODEL


def predict_views(images_rgb: list[np.ndarray], keys: list[str], model_path: Path, progress=None) -> tuple[list[View], str]:
    import torch

    model = _model(model_path)
    views, votes = [], []
    with torch.no_grad():
        for i, (img, key) in enumerate(zip(images_rgb, keys)):
            x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
            p = model(x)
            mask = _largest_island(p["mask"][0, 0].numpy() > 0.5)
            m3 = mask[..., None]
            views.append(
                View(
                    key=key,
                    rgb=img.astype(np.float64) / 255.0,
                    toc=p["toc"][0].permute(1, 2, 0).numpy().astype(np.float64) * m3,
                    normal=p["normal"][0].permute(1, 2, 0).numpy().astype(np.float64) * m3,
                    mask=mask,
                )
            )
            votes.append(int(p["footedness"][0].argmax()))
            if progress:
                progress(i + 1, len(images_rgb))
    kept = [v for v in views if v.mask.mean() >= MIN_MASK_COVERAGE]
    imgs = {k: im for k, im in zip(keys, images_rgb)}
    gains = {v.key: _paper_gain(imgs[v.key], v.mask) for v in kept}
    found = [g for g in gains.values() if g is not None]
    fallback = np.median(found, axis=0) if found else np.ones(3)
    for i, v in enumerate(kept):
        v.idx = i
        g = gains[v.key]
        v.colour = _colour_layer(imgs[v.key], v.mask, fallback if g is None else g)
    footedness = "right" if votes and np.mean(votes) >= 0.5 else "left"
    return kept, footedness


def _bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def find_matches(views: list[View], num_correspondences: int, seed: int = 10) -> Correspondences:
    rng = np.random.default_rng(seed)
    corr = Correspondences()
    per_view = num_correspondences // len(views)
    tocs, origin = [], []
    for v in views:
        interior = cv2.erode(v.mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        ys, xs = np.nonzero(interior)
        if len(xs) == 0:
            continue
        pick = rng.integers(0, len(xs), per_view)
        x = xs[pick] + rng.random(per_view) - 0.5
        y = ys[pick] + rng.random(per_view) - 0.5
        start = sum(len(t) for t in tocs)
        ids = np.arange(start, start + per_view)
        tocs.append(_bilinear(v.toc, x, y))
        origin.append(np.full(per_view, v.idx))
        corr.add(ids, origin[-1], np.stack([x, y], 1), _bilinear(v.colour, x, y), _bilinear(v.normal, x, y))
    corr.toc = np.concatenate(tocs)
    origin = np.concatenate(origin)

    d = np.arange(-SUBPIXEL_STEPS, SUBPIXEL_STEPS + 1) / SUBPIXEL_STEPS
    dx, dy = [g.ravel() for g in np.meshgrid(d, d)]
    for v in views:
        tree, pix = v.tree()
        others = np.nonzero(origin != v.idx)[0]
        dist, nn = tree.query(corr.toc[others], distance_upper_bound=5 * TOC_MATCH_MAX_DIST)
        ok = np.isfinite(dist)
        others, nn = others[ok], nn[ok]
        for s in range(0, len(others), 2000):
            cid = others[s : s + 2000]
            px, py = pix[nn[s : s + 2000]].T
            gx = px[:, None] + dx[None]
            gy = py[:, None] + dy[None]
            err = np.linalg.norm(_bilinear(v.toc, gx, gy) - corr.toc[cid][:, None], axis=-1)
            best = err.argmin(axis=1)
            keep = err[np.arange(len(cid)), best] <= TOC_MATCH_MAX_DIST
            if not keep.any():
                continue
            bx = gx[np.arange(len(cid)), best][keep]
            by = gy[np.arange(len(cid)), best][keep]
            corr.add(cid[keep], np.full(keep.sum(), v.idx), np.stack([bx, by], 1), _bilinear(v.colour, bx, by), _bilinear(v.normal, bx, by))
    return corr


def _cam_from_world(image) -> np.ndarray:
    pose = image.cam_from_world() if callable(image.cam_from_world) else image.cam_from_world
    return np.asarray(pose.matrix(), np.float64)


def calibrate(views: list[View], corr: Correspondences, workdir: Path, focal_px: float) -> dict[int, dict]:
    """Bundle-adjust camera poses on TOC matches (FOCUS `custom_matches_colmap` + `run_colmap`)."""
    import pycolmap

    if workdir.exists():
        shutil.rmtree(workdir)
    img_dir = workdir / "images"
    img_dir.mkdir(parents=True)
    names = []
    for v in views:
        name = f"{v.key}.png"
        cv2.imwrite(str(img_dir / name), cv2.cvtColor((v.rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        names.append(name)
    db_path = workdir / "database.db"
    opts = pycolmap.ImageReaderOptions()
    opts.camera_model = "SIMPLE_PINHOLE"
    opts.camera_params = f"{focal_px},{FRAME_W / 2},{FRAME_H / 2}"
    pycolmap.Database.open(db_path).close()
    pycolmap.import_images(db_path, img_dir, pycolmap.CameraMode.SINGLE, names, opts)

    c, vv, xy, _, _ = corr.arrays()
    db = pycolmap.Database.open(db_path)
    image_ids = {img.name: img.image_id for img in db.read_all_images()}
    kp_index = np.zeros(len(c), np.int64)
    for v in views:
        sel = np.nonzero(vv == v.idx)[0]
        kp_index[sel] = np.arange(len(sel))
        db.write_keypoints(image_ids[f"{v.key}.png"], (xy[sel] + 0.5).astype(np.float32))
    by_corr: dict[int, list[int]] = {}
    for i, ci in enumerate(c):
        by_corr.setdefault(int(ci), []).append(i)
    pairs: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for obs in by_corr.values():
        for a, b in combinations(obs, 2):
            va, vb = int(vv[a]), int(vv[b])
            if va == vb:
                continue
            if va > vb:
                a, b, va, vb = b, a, vb, va
            pairs.setdefault((va, vb), []).append((kp_index[a], kp_index[b]))
    lines = []
    for (va, vb), m in pairs.items():
        if len(m) < 15:
            continue
        ia, ib = image_ids[f"{views[va].key}.png"], image_ids[f"{views[vb].key}.png"]
        db.write_matches(ia, ib, np.unique(np.asarray(m, np.uint32), axis=0))
        lines.append(f"{views[va].key}.png {views[vb].key}.png")
    db.close()
    (workdir / "pairs.txt").write_text("\n".join(lines), encoding="utf-8")
    pycolmap.verify_matches(db_path, workdir / "pairs.txt")

    popts = pycolmap.IncrementalPipelineOptions()
    popts.ba_refine_focal_length = True
    popts.min_model_size = 3
    recs = pycolmap.incremental_mapping(db_path, img_dir, workdir / "sparse", popts)
    if not recs:
        return {}
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    by_name = {f"{v.key}.png": v.idx for v in views}
    out = {}
    for img in rec.images.values():
        if img.name not in by_name or not getattr(img, "has_pose", True):
            continue
        cam = rec.cameras[img.camera_id]
        K = np.asarray(cam.calibration_matrix(), np.float64)
        Rt = _cam_from_world(img)
        out[by_name[img.name]] = {"K": K, "R": Rt[:, :3], "t": Rt[:, 3], "P": K @ Rt}
    return out


def triangulate(corr: Correspondences, cams: dict[int, dict]) -> dict:
    c, vv, xy, rgb, nrm = corr.arrays()
    xy = xy + 0.5
    order = np.argsort(c, kind="stable")
    c, vv, xy, rgb, nrm = c[order], vv[order], xy[order], rgb[order], nrm[order]
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]])
    ends = np.r_[starts[1:], len(c)]
    pts, cols, normals, toc, errs = [], [], [], [], []
    for s, e in zip(starts, ends):
        obs = [i for i in range(s, e) if int(vv[i]) in cams]
        if len(obs) < 2:
            continue
        A = []
        for i in obs:
            P = cams[int(vv[i])]["P"]
            A.append(xy[i, 0] * P[2] - P[0])
            A.append(xy[i, 1] * P[2] - P[1])
        _, _, vh = np.linalg.svd(np.asarray(A))
        X = vh[-1]
        if abs(X[3]) < 1e-12:
            continue
        X = X[:3] / X[3]
        err, front, n_sum = [], True, np.zeros(3)
        for i in obs:
            cam = cams[int(vv[i])]
            pc = cam["R"] @ X + cam["t"]
            if pc[2] <= 0:
                front = False
                break
            proj = cam["K"] @ pc
            err.append(np.linalg.norm(proj[:2] / proj[2] - xy[i]))
            # FOCUS View: world normal = R_p3d @ (x, y, -z) with R_p3d = R_colmap^T diag(-1, -1, 1).
            n_sum += -cam["R"].T @ nrm[i]
        if not front:
            continue
        pts.append(X)
        # Colour away from the foot outline, where background and paper bleed in.
        inner = [i for i in obs if rgb[i, 3] >= COLOUR_EDGE_MIN_PX] or obs
        cols.append(np.median(rgb[inner, :3], axis=0))
        normals.append(n_sum / max(np.linalg.norm(n_sum), 1e-9))
        errs.append(float(np.mean(err)))
        toc.append(corr.toc[int(c[s])])
    return {
        "points": np.asarray(pts),
        "colors": np.asarray(cols),
        "normals": np.asarray(normals),
        "toc": np.asarray(toc),
        "reproj": np.asarray(errs),
    }


def _statistical_outliers(points: np.ndarray, k: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    """Open3D-style statistical outlier mask (kept = True)."""
    if len(points) <= k:
        return np.ones(len(points), bool)
    d, _ = cKDTree(points).query(points, k + 1)
    mean = d[:, 1:].mean(axis=1)
    return mean <= mean.mean() + std_ratio * mean.std()


def fuse(cloud: dict) -> dict:
    import pymeshlab
    import trimesh

    keep = cloud["reproj"] < REPROJ_MAX_PX
    pts, cols, nrm, toc = (cloud[k][keep] for k in ("points", "colors", "normals", "toc"))
    if len(pts) < 500:
        raise RuntimeError(f"Only {len(pts)} consistent 3D points")

    m = _statistical_outliers(pts, std_ratio=5.0)
    pts, cols, nrm, toc = pts[m], cols[m], nrm[m], toc[m]
    dist = np.linalg.norm(pts - pts.mean(axis=0), axis=1)
    m = dist < np.percentile(dist, 99)
    pts, cols, nrm, toc = pts[m], cols[m], nrm[m], toc[m]

    # Similarity to the template (TOC → template metres); remove a reflection as FOCUS does.
    target = toc * (TEMPLATE_MAX - TEMPLATE_MIN) + TEMPLATE_MIN
    T, _, cost = trimesh.registration.procrustes(pts, target, reflection=True, scale=True)
    if np.linalg.det(T[:3, :3]) < 0:
        T = np.diag([1.0, -1.0, 1.0, 1.0]) @ T
    P = trimesh.transform_points(pts, T)
    rot = T[:3, :3] / np.cbrt(abs(np.linalg.det(T[:3, :3])))
    N = nrm @ rot.T
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12

    zn = (P[:, 2] - TEMPLATE_MIN[2]) / (TEMPLATE_MAX[2] - TEMPLATE_MIN[2])
    m = (zn > TOC_Z_KEEP[0]) & (zn < TOC_Z_KEEP[1])
    P, cols, N = P[m], cols[m], N[m]
    m = _statistical_outliers(P, std_ratio=5.0)
    P, cols, N = P[m], cols[m], N[m]
    if np.mean(np.einsum("ij,ij->i", N, P - P.mean(axis=0))) < 0:
        N = -N

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=P, v_normals_matrix=N, v_color_matrix=np.c_[np.clip(cols, 0, 1), np.ones(len(cols))]))
    ms.generate_surface_reconstruction_screened_poisson(preclean=True, threads=1, **POISSON)
    mesh = trimesh.Trimesh(ms.current_mesh().vertex_matrix(), ms.current_mesh().face_matrix(), process=False)

    lo, hi = P.min(axis=0) - 0.001, P.max(axis=0) + 0.001
    inside = np.all((mesh.vertices >= lo) & (mesh.vertices <= hi), axis=1)
    mesh.update_faces(inside[mesh.faces].all(axis=1))
    mesh.remove_unreferenced_vertices()
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        # FOCUS `keep_pointcloud_island`: the piece the point cloud lies closest to overall.
        mesh = min(parts, key=lambda p: cKDTree(p.vertices).query(P)[0].mean())

    ms2 = pymeshlab.MeshSet()
    ms2.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
    try:
        # The sole and the ankle cut are never seen; close them so the foot reads as a solid object.
        ms2.meshing_repair_non_manifold_edges()
        ms2.meshing_close_holes(maxholesize=100000, newfaceselected=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("FOCUS hole closing skipped: %s", exc)
    if ms2.current_mesh().face_number() > TARGET_FACES:
        ms2.meshing_decimation_quadric_edge_collapse(targetfacenum=TARGET_FACES, preservenormal=True)
    mesh = trimesh.Trimesh(ms2.current_mesh().vertex_matrix(), ms2.current_mesh().face_matrix(), process=False)

    _, idx = cKDTree(P).query(mesh.vertices, k=6)
    vcol = (np.clip(cols[idx].mean(axis=1), 0, 1) * 255).astype(np.uint8)
    return {"vertices": np.asarray(mesh.vertices), "faces": np.asarray(mesh.faces), "colors": vcol, "points": P, "procrustes_cost": float(cost)}


def to_foot_frame(fused: dict, length_mm: float | None, ref_vertices: np.ndarray | None) -> dict:
    """Template frame (x heel→toe, y = foot's left, z up) → NIKONYX frame [x user right, y heel→toe, z up] in mm."""
    V = fused["vertices"].copy()
    P = fused["points"]
    z0 = np.percentile(P[:, 2], 0.5)
    low = P[P[:, 2] < z0 + 0.35 * (np.percentile(P[:, 2], 99.5) - z0)]
    xy = low[:, :2] - low[:, :2].mean(axis=0)
    w, vec = np.linalg.eigh(np.cov(xy.T))
    axis = vec[:, np.argmax(w)]
    if axis[0] < 0:
        axis = -axis
    ang = np.arctan2(axis[1], axis[0])
    c, s = np.cos(-ang), np.sin(-ang)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    V = V @ R.T
    Pr = P @ R.T
    x0, x1 = np.percentile(Pr[:, 0], [0.2, 99.8])
    mesh_len = x1 - x0
    scale = (length_mm / mesh_len) if length_mm else 1000.0
    u = (V[:, 0] - x0) * scale
    v = -(V[:, 1]) * scale
    z = np.maximum((V[:, 2] - z0) * scale, 0.0)
    out = np.column_stack([v, u, z])
    if ref_vertices is not None and len(ref_vertices):
        ref_low = ref_vertices[ref_vertices[:, 2] < 20.0]
        own_low = out[out[:, 2] < 20.0]
        if len(ref_low) and len(own_low):
            out[:, 0] += ref_low[:, 0].mean() - own_low[:, 0].mean()
    else:
        out[:, 0] -= out[out[:, 2] < 20.0, 0].mean() if np.any(out[:, 2] < 20.0) else 0.0
    return {"vertices": out, "scale_source": "a4_foot_length" if length_mm else "template_size", "mesh_length_mm": float(mesh_len * scale)}


def pick_frames(records: list[dict], root: Path, count: int = NUM_VIEWS) -> list[Path]:
    usable = [r for r in records if r.get("foot_fraction", 0) >= 0.02]
    if len(usable) < 6:
        usable = list(records)
    usable.sort(key=lambda r: r["time_s"])
    if len(usable) > count:
        usable = [usable[int(i)] for i in np.linspace(0, len(usable) - 1, count)]
    return [root / "selected" / r["filename"] for r in usable]


def run(frame_paths: list[Path], workdir: Path, model_path: Path, focal_px: float, length_mm: float | None,
        ref_vertices: np.ndarray | None, progress=None) -> dict:
    t0 = time.perf_counter()
    imgs, keys = [], []
    for i, p in enumerate(frame_paths):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        imgs.append(letterbox(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        keys.append(f"{i:03d}")
    views, footedness = predict_views(imgs, keys, model_path, progress)
    if len(views) < 4:
        raise RuntimeError(f"Foot found in only {len(views)} of {len(imgs)} frames")
    t_pred = time.perf_counter()

    cams = calibrate(views, find_matches(views, COLMAP_CORRESPONDENCES), workdir / "colmap", focal_px)
    if len(cams) < 3:
        raise RuntimeError(f"Camera positions solved for only {len(cams)} of {len(views)} frames")
    t_cal = time.perf_counter()

    cloud = triangulate(find_matches(views, FUSION_CORRESPONDENCES, seed=11), cams)
    fused = fuse(cloud)
    framed = to_foot_frame(fused, length_mm, ref_vertices)
    return {
        "vertices": framed["vertices"],
        "faces": fused["faces"],
        "colors": fused["colors"],
        "scale_source": framed["scale_source"],
        "mesh_length_mm": framed["mesh_length_mm"],
        "stats": {
            "frames": len(imgs),
            "views_with_foot": len(views),
            "views_calibrated": len(cams),
            "points": int(len(fused["points"])),
            "faces": int(len(fused["faces"])),
            "footedness_predicted": footedness,
            "procrustes_cost": round(fused["procrustes_cost"], 4),
            "seconds": {
                "prediction": round(t_pred - t0, 1),
                "calibration": round(t_cal - t_pred, 1),
                "fusion": round(time.perf_counter() - t_cal, 1),
            },
        },
    }
