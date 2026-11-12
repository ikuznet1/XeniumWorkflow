// JPEG 2000 tile decoder worker (classic script, built as its own bundle file).
//
// Dash auto-loads every .js file in assets/ as a normal page script, so everything in
// here is inert unless the script really runs inside a dedicated worker: no code path
// outside the guard touches the network, and the OpenJPEG glue (a lazily evaluated
// CommonJS module in the bundle) is only required from inside the worker.
//
// Protocol (main -> worker):
//   { type: "init", wasmUrl?, forceJs? }                -> { type: "ready", backend } | { type: "error", message }
//   { id, buffer, cropWidth?, cropHeight? } (transfer)  -> { id, data, width, height, ms } (transfer) | { id, error }
// `data` is an ArrayBuffer of little-endian uint16 samples, cropped to
// cropWidth x cropHeight (top-left corner) when those are smaller than the decoded frame.

const IN_WORKER = typeof WorkerGlobalScope !== "undefined"
  && typeof self !== "undefined"
  && self instanceof WorkerGlobalScope
  && typeof self.document === "undefined";

if (IN_WORKER) {
  let decoderPromise = null;
  let backend = "";

  const loadCodec = async ({ wasmUrl, forceJs } = {}) => {
    if (!forceJs) {
      try {
        // The .wasm is a separate asset next to this script; fetched only on first use
        // and handed over as bytes so the server's Content-Type for .wasm does not matter.
        const url = wasmUrl || new URL("openjpegwasm_decode.wasm", self.location.href).href;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP ${resp.status} for ${url}`);
        const wasmBinary = await resp.arrayBuffer();
        // eslint-disable-next-line global-require
        const factory = require("@cornerstonejs/codec-openjpeg/decodewasmjs");
        const mod = await (factory.default || factory)({ wasmBinary });
        backend = "wasm";
        return new mod.J2KDecoder();
      } catch (err) {
        if (typeof __XV_INCLUDE_JS_FALLBACK__ === "undefined" || !__XV_INCLUDE_JS_FALLBACK__) throw err;
        // eslint-disable-next-line no-console
        console.warn("[xenium-image] wasm OpenJPEG unavailable, using asm.js build:", err);
      }
    }
    if (typeof __XV_INCLUDE_JS_FALLBACK__ !== "undefined" && __XV_INCLUDE_JS_FALLBACK__) {
      // eslint-disable-next-line global-require
      const factory = require("@cornerstonejs/codec-openjpeg/decode");
      const mod = await (factory.default || factory)();
      backend = "js";
      return new mod.J2KDecoder();
    }
    throw new Error("pure-JS OpenJPEG fallback is not included in this build");
  };

  const init = (opts) => {
    if (!decoderPromise) decoderPromise = loadCodec(opts);
    return decoderPromise;
  };

  self.onmessage = async (ev) => {
    const msg = ev.data || {};
    if (msg.type === "init") {
      try {
        await init(msg);
        self.postMessage({ type: "ready", backend });
      } catch (err) {
        self.postMessage({ type: "error", message: String((err && err.message) || err) });
      }
      return;
    }
    const { id, buffer, cropWidth, cropHeight } = msg;
    try {
      const dec = await init();
      const t0 = performance.now();
      const src = new Uint8Array(buffer);
      dec.getEncodedBuffer(src.length).set(src);
      dec.decode();
      const fi = dec.getFrameInfo();
      if (fi.componentCount !== 1 || fi.bitsPerSample <= 8 || fi.bitsPerSample > 16) {
        throw new Error(`unsupported JPEG 2000 frame: ${fi.componentCount} component(s), ${fi.bitsPerSample} bit`);
      }
      const raw = dec.getDecodedBuffer(); // Uint8Array view on the codec heap (LE uint16)
      const fw = fi.width;
      const fh = fi.height;
      const w = Math.min(cropWidth || fw, fw);
      const h = Math.min(cropHeight || fh, fh);
      const out = new Uint16Array(w * h);
      // The heap view may sit at an odd offset in principle, so copy bytes, not words.
      const outBytes = new Uint8Array(out.buffer);
      if (w === fw) {
        outBytes.set(raw.subarray(0, w * h * 2));
      } else {
        for (let r = 0; r < h; r++) {
          outBytes.set(raw.subarray(r * fw * 2, (r * fw + w) * 2), r * w * 2);
        }
      }
      const ms = performance.now() - t0;
      self.postMessage({ id, data: out.buffer, width: w, height: h, ms }, [out.buffer]);
    } catch (err) {
      self.postMessage({ id, error: String((err && err.message) || err) });
    }
  };
}
