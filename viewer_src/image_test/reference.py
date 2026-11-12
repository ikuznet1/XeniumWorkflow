"""tifffile ground truth for the morphology-image layer tests.

    python reference.py png <bundle_dir> <out.png> [level] [channel]
        Downsampled greyscale PNG of one channel (percentile-stretched), for checking
        the orientation of browser screenshots.

    python reference.py tiles <tiles_dir>
        <tiles_dir>/manifest.json lists tiles dumped by verify.mjs
        ({bundle, channel, level, x, y, width, height, file}); each is compared with the
        same region read by tifffile (file opened on its own with is_ome=False).
"""
import glob
import json
import os
import sys

import numpy as np
import tifffile


def files(bundle):
    return sorted(glob.glob(os.path.join(bundle, "morphology_focus", "*.ome.tif")))


def read_tile(path, level, tx, ty):
    """Decode one TIFF tile of a pyramid level, cropped to the image extent."""
    with tifffile.TiffFile(path, is_ome=False) as t:
        page = t.series[0].levels[level].pages[0]
        h, w = page.shape
        ts = page.tilewidth
        cols = -(-w // ts)
        idx = ty * cols + tx
        fh = t.filehandle
        fh.seek(page.dataoffsets[idx])
        buf = fh.read(page.databytecounts[idx])
        arr = np.squeeze(page.decode(buf, idx)[0])
        return arr[: min(ts, h - ty * ts), : min(ts, w - tx * ts)]


def cmd_png(bundle, out, level=None, channel=0):
    from PIL import Image
    with tifffile.TiffFile(files(bundle)[int(channel)], is_ome=False) as t:
        levels = t.series[0].levels
        L = int(level) if level is not None else len(levels) - 1
        a = levels[L].asarray()
    nz = a[a > 0]
    lo, hi = (np.percentile(nz, [1, 99.9]) if nz.size else (0, 1))
    img = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(out)
    print(f"{out}: level {L} shape (h, w) = {a.shape}, 1st/99.9th percentile of non-zero = {lo:.0f}/{hi:.0f}")


def cmd_tiles(tiles_dir):
    man = json.load(open(os.path.join(tiles_dir, "manifest.json")))
    rows = []
    for m in man:
        ref = read_tile(files(m["bundle"])[m["channel"]], m["level"], m["x"], m["y"])
        got = np.fromfile(os.path.join(tiles_dir, m["file"]), dtype="<u2")
        ok_shape = ref.shape == (m["height"], m["width"]) and got.size == ref.size
        row = {k: m[k] for k in ("dataset", "channel", "level", "x", "y", "width", "height")}
        row["shape_ok"] = bool(ok_shape)
        if ok_shape:
            d = np.abs(got.reshape(ref.shape).astype(np.int64) - ref.astype(np.int64))
            row.update(max_abs_diff=int(d.max()), n_diff=int((d > 0).sum()), mean_abs_diff=float(d.mean()),
                       ref_max=int(ref.max()), ref_mean=float(ref.mean()), ref_nonzero=int((ref > 0).sum()))
        else:
            row["ref_shape"] = list(ref.shape)
        rows.append(row)
        print(json.dumps(row))
    json.dump(rows, open(os.path.join(tiles_dir, "comparison.json"), "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"png": cmd_png, "tiles": cmd_tiles}[cmd](*sys.argv[2:])
