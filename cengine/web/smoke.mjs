import createThimble from './thimble.js';
import { readFileSync } from 'fs';

const M = await createThimble();
M.FS.writeFile('/thimble-q8.bin', readFileSync('../thimble-q8.bin'));
M.FS.writeFile('/tokenizer.bin', readFileSync('../tokenizer.bin'));
const rc = M.ccall('th_init', 'number', ['string','string'], ['/thimble-q8.bin','/tokenizer.bin']);
console.log('init rc:', rc);
const catalog = readFileSync('../demo_catalog.json', 'utf8');
for (const q of ["make a reservation at Nobu for 2 people at 7pm and text Sam saying dinner is on", "whats the weather in berlin tomorrow", "sing me a happy birthday song"]) {
  const t0 = Date.now();
  const out = M.ccall('th_call', 'string', ['string','string'], [catalog, q]);
  console.log(`[${Date.now()-t0}ms] ${q.slice(0,40)} -> ${out}`);
}
