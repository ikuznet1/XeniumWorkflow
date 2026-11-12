// Decoder for the binary wire format documented in xenium_viewer_api.py:
//   uint32 header_len | header JSON (space padded to 4 bytes) | arrays, each padded to 4 bytes

const TYPES = {
  float32: Float32Array,
  uint32: Uint32Array,
  int32: Int32Array,
  uint16: Uint16Array,
  uint8: Uint8Array,
};

/** @returns {{header: object, arrays: Object<string, TypedArray>}} views into `buf` (no copies) */
export function parseBin(buf) {
  const headerLen = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headerLen)));
  const arrays = {};
  let off = 4 + headerLen;
  for (const a of header.arrays || []) {
    const T = TYPES[a.dtype];
    if (!T) throw new Error(`unsupported dtype ${a.dtype}`);
    const bytes = a.length * T.BYTES_PER_ELEMENT;
    arrays[a.name] =
      off % T.BYTES_PER_ELEMENT === 0 ? new T(buf, off, a.length) : new T(buf.slice(off, off + bytes));
    off += bytes + ((4 - (bytes % 4)) % 4);
  }
  return {header, arrays};
}

/** Encoder (used by tests / synthetic data). `arrays` is {name: TypedArray}. */
export function packBin(meta, arrays) {
  const names = Object.keys(arrays);
  const dtypeOf = (a) => Object.keys(TYPES).find((k) => a instanceof TYPES[k]);
  const listing = names.map((n) => ({name: n, dtype: dtypeOf(arrays[n]), length: arrays[n].length}));
  let head = new TextEncoder().encode(JSON.stringify({...meta, arrays: listing}));
  const padded = new Uint8Array(head.length + ((4 - (head.length % 4)) % 4)).fill(32);
  padded.set(head);
  head = padded;
  let total = 4 + head.length;
  for (const n of names) total += arrays[n].byteLength + ((4 - (arrays[n].byteLength % 4)) % 4);
  const out = new Uint8Array(total);
  new DataView(out.buffer).setUint32(0, head.length, true);
  out.set(head, 4);
  let off = 4 + head.length;
  for (const n of names) {
    const a = arrays[n];
    out.set(new Uint8Array(a.buffer, a.byteOffset, a.byteLength), off);
    off += a.byteLength + ((4 - (a.byteLength % 4)) % 4);
  }
  return out.buffer;
}
