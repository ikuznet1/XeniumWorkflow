// Optional build-time stand-in for "@vivjs/loaders" (esbuild: alias: { "@vivjs/loaders": ".../viv-loaders-shim.js" }).
// @vivjs/layers imports just these three helpers from it; the real package drags Viv's own
// OME-TIFF/Zarr loaders (geotiff 2, zarrita, zod, an LZW decoder) into the bundle, none of
// which this viewer uses. Keep in sync with @vivjs/loaders (checked against 0.23.0).
export const SIGNAL_ABORTED = "__vivSignalAborted";

export function isInterleaved(shape) {
  const lastDimSize = shape[shape.length - 1];
  return lastDimSize === 3 || lastDimSize === 4;
}

export function getImageSize(source) {
  const interleaved = isInterleaved(source.shape);
  const [height, width] = source.shape.slice(interleaved ? -3 : -2);
  return { height, width };
}
