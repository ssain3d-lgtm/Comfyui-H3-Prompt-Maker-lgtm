#!/usr/bin/env node
/**
 * The node face is a fixed-width canvas and fillText does not clip, so a long
 * model id ran straight out of the node and over whatever sat to its right.
 * LM Studio ids are repository paths and a quantised community build is well
 * over a hundred characters, so this is the normal case, not an edge one.
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const PACK = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = readFileSync(join(PACK, 'web', 'h3_maker.js'), 'utf8');
const grab = (name) => {
  const i = src.indexOf(`const ${name} = `);
  if (i < 0) throw new Error(`${name} not found in h3_maker.js`);
  return src.slice(i, src.indexOf('\n};', i) + 3);
};
const { modelLabel, fitText } = new Function(
  grab('modelLabel') + grab('fitText') + 'return { modelLabel, fitText };')();

// Stand-in metrics: 6px per latin char, 11px for the wide ones (● · …).
const ctx = { measureText: (t) => ({ width: [...t].reduce((n, c) => n + (c.charCodeAt(0) > 0x2000 ? 11 : 6), 0) }) };

let passed = 0;
const fails = [];
const ok = (n, c, d = '') => { if (c) passed++; else fails.push(`${n}${d ? `\n      ${d}` : ''}`); };
const eq = (n, a, b) => ok(n, a === b, `expected ${JSON.stringify(b)}, got ${JSON.stringify(a)}`);

// The id from the report, verbatim.
const LONG = 'aiconjured/qwen3.8-27b-uncensored-hauhaucs-aggressive-mtp-gguf-q8-nvfp4/'
  + 'qwen3.8-27b-uncensored-hauhaucs-aggressive-nvfp4-mixed.gguf';

// --- the id itself ----------------------------------------------------------
eq('label: a repository path shows only the build',
  modelLabel(LONG), 'qwen3.8-27b-uncensored-hauhaucs-aggressive-nvfp4-mixed.gguf');
eq('label: a plain id is untouched', modelLabel('qwen3-14b-instruct'), 'qwen3-14b-instruct');
eq('label: a trailing slash does not yield an empty label',
  modelLabel('org/repo/'), 'repo');
eq('label: empty falls back rather than showing nothing', modelLabel(''), '(auto)');
eq('label: null falls back', modelLabel(null), '(auto)');
eq('label: whitespace only falls back', modelLabel('   '), '(auto)');
ok('label: the extension survives — it distinguishes builds',
  modelLabel(LONG).endsWith('.gguf'));

// --- fitting ----------------------------------------------------------------
// The node's default width is 290, leaving 266 for text.
for (const room of [60, 120, 266, 400, 900]) {
  const line = `● lmstudio · ${modelLabel(LONG)}`;
  const out = fitText(ctx, line, room);
  ok(`fit: ${room}px never overflows`, ctx.measureText(out).width <= room,
    `${ctx.measureText(out).width}px > ${room}px — ${out}`);
}
{
  const short = '● lmstudio · qwen3-14b';
  eq('fit: text that already fits is returned unchanged', fitText(ctx, short, 400), short);
}
{
  // Cutting the tail would drop the quantisation, which is half of what
  // identifies the build.
  const out = fitText(ctx, `● lmstudio · ${modelLabel(LONG)}`, 266);
  ok('fit: the family survives at the head', out.includes('qwen3.8'), out);
  ok('fit: the quantisation survives at the tail', out.endsWith('.gguf'), out);
  ok('fit: the cut is marked', out.includes('…'), out);
  ok('fit: the backend name is still readable', out.startsWith('● lmstudio · '), out);
}
eq('fit: no room yields nothing rather than a stray character', fitText(ctx, 'abc', 0), '');
ok('fit: a negative width does not throw', fitText(ctx, 'abc', -5) === '');
{
  // Bisection must not walk past the ends and duplicate or drop characters.
  const text = 'abcdefghij';
  for (let w = 1; w <= 80; w++) {
    const out = fitText(ctx, text, w);
    if (ctx.measureText(out).width > w) { fails.push(`fit: overflow at ${w}px — ${out}`); break; }
    if (out.length > text.length + 1) { fails.push(`fit: grew at ${w}px — ${out}`); break; }
  }
  passed++;
}

if (fails.length) {
  console.error(`\n✗ ${fails.length} failed, ${passed} passed\n`);
  for (const f of fails) console.error('  - ' + f);
  process.exit(1);
}
console.log(`✓ all ${passed} node-face assertions passed`);
