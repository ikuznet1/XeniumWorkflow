// Optional entry point (node build.mjs --with-image): the viewer core plus the morphology-image
// module in ONE bundle, so both use the same copy of deck.gl. The image module lives in
// src/image/ and is maintained separately; its worker + .wasm are separate asset files.
import './index.js';
import * as image from './image/index.js';

if (typeof window !== 'undefined' && window.XeniumViewer) window.XeniumViewer.image = image;
