// Small pool of JPEG 2000 decode workers. Each worker owns one initialised OpenJPEG
// instance; jobs are queued on the main thread so that a job whose tile has scrolled
// out of view can be dropped before it ever reaches a worker.

export const ABORTED = "__vivSignalAborted"; // same sentinel Viv uses (SIGNAL_ABORTED)

export function defaultPoolSize() {
  const hc = (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 4;
  return Math.max(1, Math.min(6, hc - 1));
}

export class DecoderPool {
  /**
   * @param {object} opts
   * @param {string} opts.workerUrl  URL of the worker bundle (xenium_image_worker.js)
   * @param {string} [opts.wasmUrl]  URL of openjpegwasm_decode.wasm (default: next to the worker)
   * @param {number} [opts.size]
   * @param {boolean} [opts.forceJs] use the asm.js codec (only if the worker bundle includes it)
   */
  constructor({ workerUrl, wasmUrl, size, forceJs = false }) {
    this.workerUrl = workerUrl;
    this.wasmUrl = wasmUrl;
    this.forceJs = forceJs;
    this.size = size || defaultPoolSize();
    this.workers = []; // { worker, busy, ready: Promise, job }
    this.queue = [];
    this.nextId = 1;
    this.destroyed = false;
    this.backend = null;
    this.decodeTimes = []; // ms spent inside the codec, most recent 500
  }

  _spawn() {
    const worker = new Worker(this.workerUrl);
    const slot = { worker, job: null, ready: null, failed: null };
    slot.ready = new Promise((resolve, reject) => {
      const onInit = (ev) => {
        const m = ev.data || {};
        if (m.type === "ready") {
          this.backend = m.backend;
          worker.removeEventListener("message", onInit);
          resolve();
        } else if (m.type === "error") {
          worker.removeEventListener("message", onInit);
          reject(new Error(`JPEG 2000 worker failed to initialise: ${m.message}`));
        }
      };
      worker.addEventListener("message", onInit);
      worker.addEventListener("error", (e) => {
        const err = new Error(`JPEG 2000 worker error: ${e.message || this.workerUrl}`);
        slot.failed = err;
        reject(err);
        if (slot.job && slot.job.started) {
          slot.job.reject(err);
          slot.job = null;
        }
      });
    });
    slot.ready.catch((err) => { slot.failed = err; });
    worker.addEventListener("message", (ev) => {
      const m = ev.data || {};
      if (m.id === undefined) return;
      const job = slot.job;
      if (!job || job.id !== m.id) return;
      slot.job = null;
      if (m.error) job.reject(new Error(m.error));
      else {
        this.decodeTimes.push(m.ms);
        if (this.decodeTimes.length > 500) this.decodeTimes.shift();
        job.resolve({ data: new Uint16Array(m.data), width: m.width, height: m.height, ms: m.ms });
      }
      this._pump();
    });
    worker.postMessage({ type: "init", wasmUrl: this.wasmUrl, forceJs: this.forceJs });
    this.workers.push(slot);
    return slot;
  }

  _pump() {
    if (this.destroyed) return;
    while (this.queue.length) {
      let slot = this.workers.find((s) => !s.job && !s.failed);
      if (!slot && this.workers.length < this.size) slot = this._spawn();
      if (!slot) {
        if (this.workers.length && this.workers.every((s) => s.failed)) {
          const err = this.workers[0].failed;
          for (const job of this.queue.splice(0)) job.reject(err);
        }
        return;
      }
      const job = this.queue.shift();
      slot.job = job;
      slot.ready.then(() => {
        if (job.cancelled) { // aborted while the worker was still starting up
          slot.job = null;
          this._pump();
          return;
        }
        job.started = true;
        slot.worker.postMessage(
          { id: job.id, buffer: job.buffer, cropWidth: job.cropWidth, cropHeight: job.cropHeight },
          [job.buffer],
        );
      }, (err) => {
        slot.job = null;
        job.reject(err);
        this._pump();
      });
    }
  }

  /**
   * Decode one JPEG 2000 codestream to uint16 samples.
   * @param {ArrayBuffer} buffer compressed bytes (transferred to the worker)
   * @param {{cropWidth?:number, cropHeight?:number, signal?:AbortSignal}} [opts]
   * @returns {Promise<{data:Uint16Array,width:number,height:number,ms:number}>}
   */
  decode(buffer, { cropWidth, cropHeight, signal } = {}) {
    if (this.destroyed) return Promise.reject(new Error("decoder pool destroyed"));
    if (signal && signal.aborted) return Promise.reject(ABORTED);
    return new Promise((resolve, reject) => {
      const job = { id: this.nextId++, buffer, cropWidth, cropHeight, resolve, reject, started: false, cancelled: false };
      if (signal) {
        signal.addEventListener("abort", () => {
          if (job.started) return; // already inside a worker: let it finish (a few tens of ms)
          job.cancelled = true;
          const i = this.queue.indexOf(job);
          if (i >= 0) this.queue.splice(i, 1);
          reject(ABORTED);
        }, { once: true });
      }
      this.queue.push(job);
      this._pump();
    });
  }

  destroy() {
    this.destroyed = true;
    for (const s of this.workers) {
      s.worker.terminate();
      if (s.job) s.job.reject(ABORTED);
    }
    for (const job of this.queue) job.reject(ABORTED);
    this.workers = [];
    this.queue = [];
  }
}
