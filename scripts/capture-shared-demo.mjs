// Exercises the real S3-backed viewer, including playback, and records evidence outside git.
import { mkdir, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright-core';
import path from 'node:path';
import assert from 'node:assert/strict';
const out=process.argv[2];
if(!out)throw new Error('Usage: node scripts/capture-shared-demo.mjs /absolute/evidence/directory');
await mkdir(out,{recursive:true});
const base='http://127.0.0.1:5399';
const status=await (await fetch(`${base}/api/shared-assets`)).json();
assert.equal(status.mode,'s3','Test requires remote mode; local assets must not mask failures');
const results={status,clips:{}};
for(const clip of ['elevator','lobby','stairs2','atrium','tos31']){
 const browser=await chromium.launch({channel:'chrome',headless:true,args:['--use-angle=metal','--ignore-gpu-blocklist']});
 try{
  const context=await browser.newContext({viewport:{width:1440,height:900}});
  const page=await context.newPage();const errors=[],failed=[],remote=[];
  page.on('pageerror',e=>errors.push(e.message));
  page.on('response',r=>{if(r.status()>=400)failed.push({url:r.url(),status:r.status()});if(r.headers()['x-wander-asset-source']==='s3')remote.push(r.url());});
  const begin=performance.now();
  await page.goto(`${base}/demo.html?clip=${clip}&walk=1`,{waitUntil:'domcontentloaded'});
  await page.waitForFunction(()=>{const f=document.querySelector('iframe');return f?.contentWindow?.wander?.ready && !document.querySelector('#play')?.disabled;},null,{timeout:300000});
  const readySeconds=(performance.now()-begin)/1000;
  const frame=page.frames().find(f=>f.url().includes('/fourd.html'));
  assert.equal(await frame.evaluate(()=>window.wander.demo),clip,'Wrapper must load the requested preset');
  const initial=await frame.evaluate(()=>window.wander.playing);
  await page.locator('body').click({position:{x:700,y:20}});
  await page.keyboard.press('Enter');
  await page.waitForTimeout(500);
  const toggled=await frame.evaluate(()=>window.wander.playing);
  assert.equal(toggled,!initial);
  await page.keyboard.press('Enter');
  assert.equal(await frame.evaluate(()=>window.wander.playing),initial);
  await frame.evaluate(()=>{window.wander.play(false);});
  await page.waitForTimeout(500);
  await page.screenshot({path:path.join(out,`${clip}-s3.png`)});
  const state=await frame.evaluate(()=>({people:window.wander.people.map(p=>({loaded:p.loaded,frames:p.nF})),walk:!!window.wander.walk}));
  results.clips[clip]={readySeconds,state,remoteRequests:remote.length,errors,failed,enterToggle:true};
  assert.ok(remote.some(u=>u.includes('/worlds/')));assert.ok(remote.some(u=>u.includes('.spz')));
  assert.equal(errors.length,0,JSON.stringify(errors));
  console.log(JSON.stringify({clip,...results.clips[clip]}));
  await writeFile(path.join(out,'report.json'),JSON.stringify(results,null,2));
  await context.close();
 }finally{await browser.close();}
}
