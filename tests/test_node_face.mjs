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

// --- the host bridge boundary ------------------------------------------------
// The overlay half (platform.ts) has always checked the origin and the sending
// window. The host half checked only the `source` string, which any sender can
// write — so any frame holding a handle to the ComfyUI window could post an
// "apply" while the overlay was open and overwrite the node's result widget,
// which is what feeds the downstream sockets. Pinned as source text because
// the listener closes over module state that cannot be stood up here.
const listener = src.slice(src.indexOf('window.addEventListener("message"'));
ok('bridge: the sender origin is checked',
  /e\.origin\s*!==\s*window\.location\.origin/.test(listener), listener.slice(0, 200));
ok('bridge: the sending window is checked, not just the source string',
  /e\.source\s*!==\s*overlay\?\.frame\?\.contentWindow/.test(listener));
ok('bridge: both checks come before the payload is trusted',
  listener.indexOf('e.origin') < listener.indexOf('d.type'));
ok('bridge: outbound messages name a concrete target origin, not "*"',
  /postMessage\(\s*\{[^}]*\},\s*window\.location\.origin\s*\)/s.test(src)
  && !/postMessage\([^)]*,\s*"\*"\s*\)/s.test(src));

// --- the Instant node shares the extension ----------------------------------
// One extension draws both nodes. If the registration check ever goes back to
// a single name, the Instant node comes up as three bare text widgets and no
// buttons — it works, but nobody can open the overlay to fill it in.
ok('instant: the extension registers for the Instant node too',
  /nodeData\?\.name !== NODE && nodeData\?\.name !== NODE_INSTANT/.test(src));
ok('instant: its face describes the Queue behaviour, not an applied result',
  /summarizeInstant/.test(src) && /Queue마다 생성/.test(src));

// --- saved-settings migration -------------------------------------------------
// readLlm rewrites every pre-optimization workflow on first open. It had no
// test at all, and it is the one function here that mutates the user's file.
const { readLlm, DEFAULT_LLM } = new Function(
  'const readJson = (node, name, fb) => node.__llm ?? fb;'
  + 'const writeJson = (node, name, value) => { node.__written = value; };'
  + src.slice(src.indexOf('const DEFAULT_LLM = '), src.indexOf('\n};', src.indexOf('const DEFAULT_LLM = ')) + 3)
  + grab('readLlm')
  + 'return { readLlm, DEFAULT_LLM };')();

eq('migrate: a fresh node with nothing saved gets the fast defaults',
  [readLlm({}).thinking, readLlm({}).unload_after, readLlm({}).prompt_profile].join(','),
  'off,close,fast');
ok('migrate: an empty node is not written back to — nothing changed',
  new Function('const readJson=(n,k,f)=>f;const writeJson=(n,k,v)=>{n.__written=v;};'
    + src.slice(src.indexOf('const DEFAULT_LLM = '), src.indexOf('\n};', src.indexOf('const DEFAULT_LLM = ')) + 3)
    + grab('readLlm') + 'const n={};readLlm(n);return n.__written===undefined;')());

const old1 = { __llm: { backend: 'lmstudio', thinking: 'auto', unload_after: 'now', max_tokens: 60000 } };
const migrated = readLlm(old1);
eq('migrate: a v1 workflow moves off the slow defaults',
  [migrated.thinking, migrated.unload_after, migrated.prompt_profile].join(','), 'off,close,fast');
eq('migrate: it is stamped so it never migrates twice', migrated.settings_version, 2);
ok('migrate: the change is written back to the widget', old1.__written !== undefined);
eq('migrate: unrelated saved values survive', migrated.max_tokens, 60000);

const v2 = { __llm: { settings_version: 2, thinking: 'auto', unload_after: 'now', prompt_profile: 'full' } };
const kept = readLlm(v2);
eq('migrate: a deliberate post-release choice of auto is never overwritten', kept.thinking, 'auto');
eq('migrate: ...nor a deliberate 즉시 언로드', kept.unload_after, 'now');
eq('migrate: ...nor a deliberate Full profile', kept.prompt_profile, 'full');
ok('migrate: a v2 node is left alone on disk', v2.__written === undefined);

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
