// Simulated WebXR smoke check; this does not measure Quest frame rate.
import { chromium } from 'playwright-core';
const B='http://127.0.0.1:5399';
const browser = await chromium.launch({ channel:'chrome', headless:false, args:['--autoplay-policy=no-user-gesture-required'] });
try {
const page = await browser.newPage({ viewport:{width:1200,height:800} });
await page.addInitScript({ path: new URL('./fake-webxr.js', import.meta.url).pathname });
const errs=[]; page.on('pageerror',e=>errs.push('pageerror: '+e.message)); page.on('console',m=>{if(m.type()==='error')errs.push('c: '+m.text())});
await page.goto(`${B}/fourd.html?demo=tos31&xr=1&fakexr=1&xradapt=0`, {waitUntil:'load',timeout:120000});
await page.waitForFunction(()=>window.wander?.ready===true,null,{timeout:180000});
await page.waitForTimeout(1500);
console.log('pre', JSON.stringify(await page.evaluate(()=>({ btn: document.getElementById('xrBtn').style.display, hasBinding: typeof XRWebGLBinding, fake: !!window.__fakeXR }))));
await page.click('#xrBtn');
await page.waitForTimeout(4000);
const post = await page.evaluate(()=>({
  presenting: window.wander.spark.renderer.xr.isPresenting,
  frames: window.__fakeXR.frames.length,
  canvas:[window.wander.spark.renderer.domElement.width, window.wander.spark.renderer.domElement.height],
  camParent: window.wander.camera.parent?.name ?? null,
  xr: window.__xr, hud: document.getElementById('xrHud').textContent,
}));
console.log('post', JSON.stringify(post, null, 1));
console.log('errors', errs.slice(0,8));
} finally { await browser.close(); }
