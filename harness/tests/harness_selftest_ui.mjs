#!/usr/bin/env node
// UI-level checks for Stirrup (needs Playwright's Chromium). Usage: node tests/harness_selftest_ui.mjs [base-url]
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
function loadPlaywright() {
  try { return require('playwright'); } catch (e) {}
  const home = process.env.HOME || '';
  const bases = [path.join(home, '.local/share/mise/installs/npm-playwright')];
  for (const base of bases) {
    if (!fs.existsSync(base)) continue;
    for (const v of fs.readdirSync(base).sort().reverse()) {
      for (const pkg of ['playwright', 'playwright-core']) {
        const p = path.join(base, v, 'node_modules', pkg);
        if (fs.existsSync(p)) return require(p);
      }
    }
  }
  throw new Error('playwright not found');
}
const BASE = (process.argv[2] || 'http://127.0.0.1:8090').replace(/\/$/, '');
const sleep = ms => new Promise(r => setTimeout(r, ms));
const results = [];
const rec = (name, ok, detail = '') => {
  results.push({ name, ok: !!ok, detail: String(detail).slice(0, 140) });
  console.log((ok ? 'PASS  ' : 'FAIL  ') + name + '  ' + String(detail).slice(0, 110));
};
async function until(fn, ms, step = 250) {
  const end = Date.now() + ms;
  while (Date.now() < end) { try { if (await fn()) return true; } catch (e) {} await sleep(step); }
  return false;
}

(async () => {
  const { chromium } = loadPlaywright();
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 430, height: 900 } });
  const page = await ctx.newPage();
  const errs = [];
  page.on('pageerror', e => errs.push('pageerror:' + e.message));
  page.on('console', m => { if (m.type() === 'error') errs.push('console:' + m.text()); });
  await page.goto(BASE + '/');
  await page.waitForTimeout(2200);

  rec('page title', (await page.title()) === 'Stirrup', await page.title());
  rec('manifest name', (await page.evaluate(async () => (await (await fetch('/manifest.webmanifest')).json()).name)) === 'Stirrup', 'Stirrup');
  await page.evaluate(() => openDrawer()); await page.waitForTimeout(400);
  rec('drawer brand', (await page.evaluate(() => (document.querySelector('.dbrand .bname') || {}).textContent)) === 'Stirrup', 'Stirrup');
  const history = await page.evaluate(() => ({ groups: document.querySelectorAll('#sessionList .sgroup').length, rows: document.querySelectorAll('#sessionList .sess').length }));
  rec('history list', history.groups >= 1 && history.rows >= 1, `${history.groups} groups, ${history.rows} rows`);
  await page.evaluate(() => closeDrawer()); await page.waitForTimeout(300);

  // one session for chat + agent, deleted afterwards
  await page.evaluate(() => addSession()); await page.waitForTimeout(400);
  const sid = await page.evaluate(() => STORE.activeId);
  if (!(await page.evaluate(() => document.querySelector('#thinkChip').classList.contains('on')))) await page.click('#thinkChip');
  await page.fill('#input', 'In one sentence: why is the sky blue?');
  await page.press('#input', 'Enter');
  let sawThink = false;
  const t0 = Date.now();
  while (Date.now() - t0 < 120000) {
    const st = await page.evaluate(() => ({ done: ['ok', 'err'].includes(document.querySelector('#status').className), think: !!document.querySelector('.think-row') }));
    if (st.think) sawThink = true;
    if (st.done) break;
    await sleep(250);
  }
  const answer = await page.evaluate(() => (document.querySelector('.abody') || {}).innerText || '');
  rec('chat answer', /blue|Rayleigh|scatter/i.test(answer), answer.slice(0, 80).replace(/\n/g, ' '));
  rec('one-line thinking', sawThink, sawThink ? 'seen live' : 'not seen');
  await page.evaluate(() => openStatus()); await page.waitForTimeout(350);
  const usage = await page.evaluate(() => document.querySelector('#statusBody').innerText);
  rec('usage panel', /Usage/i.test(usage) && /last turn/i.test(usage), 'token/cache block present');
  await page.evaluate(() => hideSheets()); await page.waitForTimeout(200);

  // agent round-trip
  if (!(await page.evaluate(() => document.querySelector('#agentChip').classList.contains('on')))) await page.click('#agentChip');
  if (await page.evaluate(() => document.querySelector('#thinkChip').classList.contains('on'))) await page.click('#thinkChip');
  const before = await page.evaluate(() => document.querySelectorAll('.msg.assistant').length);
  const MODELS = process.env.STIRRUP_MODELS || require('node:path').join(process.env.HOME || '', 'stirrup', 'models');
  await page.fill('#input', 'Count the .gguf files in ' + MODELS + ' and answer with just the number.');
  await page.press('#input', 'Enter');
  await until(async () => (await page.evaluate(() => document.querySelectorAll('.msg.assistant').length)) > before ||
                          (await page.evaluate(() => !document.querySelector('#stop').hidden)), 10000, 200);
  const agentDone = await until(async () => ['ok', 'err'].includes(await page.evaluate(() => document.querySelector('#status').className)), 180000, 300);
  const agent = await page.evaluate(() => {
    const a = [...document.querySelectorAll('.msg.assistant')].pop();
    return { answer: (a.querySelector('.abody') || {}).innerText || '', steps: (a.querySelector('.steps') || {}).innerText || '' };
  });
  rec('agent round-trip', agentDone && /^\s*\d+\s*$/.test(agent.answer.trim()) && /step/.test(agent.steps),
      'answer=' + agent.answer.trim().slice(0, 12) + ', steps=' + agent.steps.split('\n')[0].slice(0, 30));
  rec('console clean', errs.length === 0, JSON.stringify([...new Set(errs)]).slice(0, 120));

  // cleanup the throwaway session (and its synced copy)
  await page.evaluate((id) => { if (id) deleteSession(id); }, sid);
  await page.waitForTimeout(1200);
  const passed = results.filter(x => x.ok).length;
  const out = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..', 'results', 'selftest-ui-last.json');
  try { fs.writeFileSync(out, JSON.stringify({ at: new Date().toISOString(), base: BASE, results }, null, 2)); } catch (e) {}
  console.log(`\nStirrup UI self-test: ${passed}/${results.length} passed`);
  await browser.close();
  process.exit(passed === results.length ? 0 : 1);
})().catch(e => { console.error('FATAL', e); process.exit(1); });
