// Viewport-driven loader for data that is fetched per visible region (boundaries, transcripts).
// Keeps one padded region in memory, refetches only when the viewport leaves it, and
// cancels in-flight requests so a stale response can never overwrite a newer one.

const contains = (outer, inner) =>
  outer[0] <= inner[0] && outer[1] <= inner[1] && outer[2] >= inner[2] && outer[3] >= inner[3];

export class RegionLoader {
  /**
   * @param {(url: string, signal: AbortSignal) => Promise<{header, arrays}>} fetchBin
   * @param {() => void} onChange called whenever data / status changed
   * @param {number} pad fraction of the viewport size added on every side
   */
  constructor(fetchBin, onChange, pad = 0.25) {
    this.fetchBin = fetchBin;
    this.onChange = onChange;
    this.pad = pad;
    this.seq = 0;
    this.requests = 0; // number of requests issued (for tests)
    this._reset();
  }

  _reset() {
    if (this.ctrl) this.ctrl.abort();
    this.ctrl = null;
    this.seq++;
    this.key = null;
    this.box = null; // region covered by `data` or by the in-flight request
    this.data = null; // {header, arrays} of the last complete, non-truncated response
    this.info = null; // header of the last response
    this.truncBox = null; // smallest region known to be truncated
    this.refinable = false; // response was subsampled: zooming in can fetch more detail
    this.fetchW = 0;
    this.loading = false;
    this.error = null;
  }

  get active() {
    return this.key !== null;
  }

  /**
   * @param {number[]|null} vp [x0, y0, x1, y1] current viewport
   * @param {{enabled: boolean, key: string, url: (box: number[]) => string}} spec
   */
  update(vp, spec) {
    if (!spec.enabled) {
      if (this.active) {
        this._reset();
        this.onChange();
      }
      return;
    }
    if (spec.key !== this.key) {
      this._reset();
      this.key = spec.key;
      this.onChange();
    }
    if (!vp || !(vp[2] > vp[0]) || !(vp[3] > vp[1])) return;
    if (this.box && contains(this.box, vp) && !(this.refinable && vp[2] - vp[0] < 0.7 * this.fetchW)) return;
    if (this.truncBox && contains(vp, this.truncBox)) return; // a superset is truncated too
    // Padding only makes sense if the padded box is clearly smaller than a box the server
    // already refused; otherwise ask for the exact view straight away.
    const k = 1 + 2 * this.pad;
    const t = this.truncBox;
    const padded = !(t && (vp[2] - vp[0]) * (vp[3] - vp[1]) * k * k >= 0.8 * (t[2] - t[0]) * (t[3] - t[1]));
    this._start(vp, spec, padded);
  }

  _start(vp, spec, padded) {
    if (this.ctrl) this.ctrl.abort();
    const ctrl = (this.ctrl = new AbortController());
    const seq = ++this.seq;
    const px = padded ? (vp[2] - vp[0]) * this.pad : 0;
    const py = padded ? (vp[3] - vp[1]) * this.pad : 0;
    const box = [vp[0] - px, vp[1] - py, vp[2] + px, vp[3] + py];
    this.box = box;
    this.fetchW = vp[2] - vp[0];
    this.loading = true;
    this.error = null;
    this.requests++;
    this.onChange();
    this.fetchBin(spec.url(box), ctrl.signal).then(
      (res) => {
        if (seq !== this.seq) return;
        if (res.header.truncated && padded) {
          // The padding may be what pushed us over the server limit; retry the exact view.
          this._start(vp, spec, false);
          return;
        }
        this.info = res.header;
        if (res.header.truncated) {
          this.truncBox = box;
          this.box = null;
          this.data = null;
          this.refinable = false;
        } else {
          this.data = res;
          this.truncBox = null;
          this.refinable = !!res.header.subsampled;
        }
        this.loading = false;
        this.onChange();
      },
      (err) => {
        if (seq !== this.seq || (err && err.name === 'AbortError')) return;
        this.loading = false;
        this.box = null;
        this.error = String((err && err.message) || err);
        this.onChange();
      }
    );
  }

  destroy() {
    this._reset();
  }
}
