// Header-only access to a Xenium morphology OME-TIFF: geotiff.js parses the (Big)TIFF
// IFDs over HTTP Range requests; the pyramid levels live in the SubIFDs of IFD 0.
// Every file is treated as an independent single-channel pyramid (in the multi-file
// layout each file carries OME-XML for all channels but pixel data for one only).

import { fromUrl } from "geotiff";

export const COMPRESSION_JPEG2000 = 34712; // "Aperio"-style JPEG 2000 tag used by Xenium

const TAGS = ["ImageWidth", "ImageLength", "TileWidth", "TileLength", "Compression",
  "BitsPerSample", "SamplesPerPixel", "TileOffsets", "TileByteCounts"];

async function readLevel(ifd) {
  if (!ifd.hasTag("TileOffsets")) throw new Error("morphology image is not tiled");
  const [width, height, tileWidth, tileHeight, compression, bps, spp, offsets, byteCounts] = await Promise.all(
    TAGS.map((t) => (ifd.hasTag(t) ? ifd.loadValue(t) : undefined)),
  );
  const bits = (bps && bps.length !== undefined) ? bps[0] : bps;
  return {
    width: Number(width),
    height: Number(height),
    tileWidth: Number(tileWidth),
    tileHeight: Number(tileHeight),
    compression: Number(compression),
    bitsPerSample: Number(bits),
    samplesPerPixel: Number(spp || 1),
    cols: Math.ceil(Number(width) / Number(tileWidth)),
    rows: Math.ceil(Number(height) / Number(tileHeight)),
    // Float64 holds byte offsets exactly up to 2^53, far beyond any slide file.
    offsets: Float64Array.from(offsets, Number),
    byteCounts: Float64Array.from(byteCounts, Number),
  };
}

/** Channel names from OME-XML, plus which file holds which channel (multi-file layout). */
export function parseOme(xml) {
  const out = { channelNames: [], fileChannel: {}, physicalSizeX: null, sizeC: null };
  if (!xml || xml.indexOf("<OME") < 0) return out;
  const attr = (tag, name) => {
    const m = new RegExp(`\\b${name}="([^"]*)"`).exec(tag);
    return m ? m[1] : null;
  };
  const unescape = (s) => s.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'").replace(/&amp;/g, "&");
  for (const m of xml.matchAll(/<Channel\b[^>]*>/g)) {
    const name = attr(m[0], "Name");
    out.channelNames.push(name === null ? null : unescape(name));
  }
  const px = /<Pixels\b[^>]*>/.exec(xml);
  if (px) {
    const ps = attr(px[0], "PhysicalSizeX");
    const sc = attr(px[0], "SizeC");
    if (ps !== null) out.physicalSizeX = Number(ps);
    if (sc !== null) out.sizeC = Number(sc);
  }
  for (const m of xml.matchAll(/<TiffData\b([^>]*)>\s*<UUID\b([^>]*)>/g)) {
    const file = attr(m[2], "FileName");
    const c = attr(m[1], "FirstC");
    if (file !== null) out.fileChannel[file] = c === null ? 0 : Number(c);
  }
  return out;
}

/**
 * Open one file: IFD 0, its SubIFD pyramid and the tile index of every level.
 * @returns {Promise<{url:string, levels:object[], ome:object, openMs:number}>}
 */
export async function openPyramid(url, { signal } = {}) {
  const t0 = performance.now();
  // Small blocks: the IFDs and tile index are a few kB each, scattered through the file.
  const tiff = await fromUrl(url, { blockSize: 16384, cacheSize: 64 }, signal);
  const image = await tiff.getImage(0);
  const ifd0 = image.fileDirectory;
  const [subOffsets, description] = await Promise.all([
    ifd0.hasTag("SubIFDs") ? ifd0.loadValue("SubIFDs") : [],
    ifd0.hasTag("ImageDescription") ? ifd0.loadValue("ImageDescription") : "",
  ]);
  const subIfds = await Promise.all(
    Array.from(subOffsets || [], (o) => tiff.parser.parseFileDirectoryAt(Number(o))),
  );
  const levels = await Promise.all([ifd0, ...subIfds].map(readLevel));
  // Keep only a properly decreasing pyramid (SubIFDs could in principle hold other things).
  const pyramid = [levels[0]];
  for (let i = 1; i < levels.length; i++) {
    const prev = pyramid[pyramid.length - 1];
    const lv = levels[i];
    const halved = (a, b) => b === Math.floor(a / 2) || b === Math.ceil(a / 2);
    if (lv.tileWidth === levels[0].tileWidth && lv.tileHeight === levels[0].tileHeight
      && halved(prev.width, lv.width) && halved(prev.height, lv.height)) pyramid.push(lv);
    else break;
  }
  const l0 = pyramid[0];
  if (l0.compression !== COMPRESSION_JPEG2000) {
    throw new Error(`${url}: TIFF compression ${l0.compression} is not supported (expected JPEG 2000, 34712)`);
  }
  if (l0.samplesPerPixel !== 1 || l0.bitsPerSample !== 16) {
    throw new Error(`${url}: expected 1 x uint16 sample per pixel, got ${l0.samplesPerPixel} x ${l0.bitsPerSample} bit`);
  }
  if (l0.tileWidth !== l0.tileHeight) throw new Error(`${url}: non-square tiles are not supported`);
  return {
    url,
    levels: pyramid,
    ome: parseOme(typeof description === "string" ? description : ""),
    openMs: performance.now() - t0,
  };
}
