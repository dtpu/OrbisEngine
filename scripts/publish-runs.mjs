#!/usr/bin/env bun
// Immutable, checksum-verified objects first; the shared pointer is updated LAST.
import { mkdir, readFile, writeFile, stat } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { parseArgs } from 'node:util';
import { PutObjectCommand, ListObjectsV2Command } from '@aws-sdk/client-s3';
import { ROOT, config, client, getJSON, hashFile, filesUnder, mime } from './lib/shared-storage.mjs';
const {values:args} = parseArgs({options:{'evidence-dir':{type:'string'},'assets-only':{type:'boolean'},concurrency:{type:'string',default:'10'}}});
const concurrency = Number(args.concurrency);
if(!Number.isInteger(concurrency)||concurrency<1||concurrency>32) throw new Error('concurrency must be 1–32');
const s3 = client(); // Author's AWS profile, NEVER the teammate credentials in .env.local.
const cacheDir=path.join(ROOT,'.context/shared-storage'); await mkdir(cacheDir,{recursive:true});
const cacheFile=path.join(cacheDir,'upload-cache.json');
let cache={}; try{cache=JSON.parse(await readFile(cacheFile,'utf8'));}catch{}
let previous={value:{files:{},archives:{}},etag:null};
try { previous=await getJSON(s3,config.catalogKey); } catch(e) {if(e.name!=='NoSuchKey' && e.$metadata?.httpStatusCode!==404) throw e;}
let base={files:{},archives:{}};
if(previous.value.snapshot) base=(await getJSON(s3,previous.value.snapshot)).value;
if(previous.value.archive) base.archives=(await getJSON(s3,previous.value.archive)).value.files;
const known=new Set();
for (const prefix of ['viewer/blobs/','archive/blobs/']) {
  let token;
  do {
    const r=await s3.send(new ListObjectsV2Command({Bucket:config.bucket,Prefix:prefix,ContinuationToken:token}));
    for(const x of r.Contents||[]) known.add(x.Key);
    token=r.NextContinuationToken;
  } while(token);
}
const publicScan=await filesUnder(path.join(ROOT,'public'),'/');
const tasks=publicScan.files.map(x=>({...x,kind:'files',prefix:'viewer'}));
const exclusions={public:publicScan.excluded};
if(!args['assets-only']) {
  const runScan=await filesUnder(path.join(ROOT,'.context/run'),'runs/');
  tasks.push(...runScan.files.map(x=>({...x,kind:'archives',prefix:'archive'})));
  exclusions.runs=runScan.excluded;
  if(args['evidence-dir']) {
    const scan=await filesUnder(path.resolve(args['evidence-dir']),'evidence/');
    tasks.push(...scan.files.map(x=>({...x,kind:'archives',prefix:'archive'})));
    exclusions.evidence=scan.excluded;
  }
}
console.log(`Inventory: ${tasks.length} files, ${(tasks.reduce((s,t)=>s+t.size,0)/1e9).toFixed(2)} GB; ${known.size} existing blobs. Exclusions: ${Object.values(exclusions).flat().length}.`);
await writeFile(path.join(cacheDir,'excluded.json'),JSON.stringify(exclusions,null,2));
const next={schema:'wander.shared/1',createdAt:new Date().toISOString(),files:{...base.files},archives:{...base.archives}};
let cursor=0,done=0,uploaded=0,bytes=0,failed,advanced=false;
const inFlight=new Map();
const progress=setInterval(()=>console.log(`Published blobs ${done}/${tasks.length}; new ${uploaded}, ${(bytes/1e9).toFixed(2)} GB uploaded`),15000);
async function worker(){
 while(cursor<tasks.length && !failed){
  const t=tasks[cursor++];
  try{
    const old=cache[t.file];
    const sha=old?.size===t.size && old?.mtimeMs===t.mtimeMs ? old.sha256 : await hashFile(t.file);
    const key=`${t.prefix}/blobs/${sha}`;
    if(!known.has(key)) {
      if(!inFlight.has(key)) inFlight.set(key,(async()=>{
        if(t.size>5_000_000_000) throw new Error(`File exceeds single-object uploader limit: ${t.path}`);
        for(let attempt=0;;attempt++) {
          try {
            await s3.send(new PutObjectCommand({Bucket:config.bucket,Key:key,Body:createReadStream(t.file),ContentLength:t.size,ContentType:mime(t.path),ChecksumSHA256:Buffer.from(sha,'hex').toString('base64'),Metadata:{sha256:sha},IfNoneMatch:'*'}));
            break;
          } catch(e) {
            if(e.$metadata?.httpStatusCode===412) break;
            if(attempt>=5 || (e.$metadata?.httpStatusCode<500 && ![408,409,429].includes(e.$metadata?.httpStatusCode))) throw e;
            await new Promise(r=>setTimeout(r,Math.min(1000*2**attempt,16000)));
          }
        }
        known.add(key);uploaded++;bytes+=t.size;
      })());
      await inFlight.get(key);
    }
    const after=await stat(t.file);
    if(after.size!==t.size || after.mtimeMs!==t.mtimeMs) throw new Error(`File changed during publish; retry after generation finishes: ${t.path}`);
    next[t.kind][t.path]={key,size:t.size,sha256:sha,contentType:mime(t.path)};
    cache[t.file]={size:t.size,mtimeMs:t.mtimeMs,sha256:sha};done++;
  }catch(e){failed=e;}
 }
}
try{
 await Promise.all(Array.from({length:concurrency},worker));
 if(failed) throw failed;
 // Recheck the entire inventory: an active pipeline must not publish a mixed snapshot.
 for(const t of tasks){const s=await stat(t.file);if(s.size!==t.size||s.mtimeMs!==t.mtimeMs) throw new Error(`Changed during publish: ${t.path}; retry`);}
 const id=randomUUID();
 const archiveKey=`archive/snapshots/${id}.json`,snapshot=`viewer/snapshots/${id}.json`;
 const putJSON=(Key,value,condition={})=>s3.send(new PutObjectCommand({Bucket:config.bucket,Key,Body:JSON.stringify(value),ContentType:'application/json',...condition}));
 await putJSON(archiveKey,{schema:next.schema,createdAt:next.createdAt,files:next.archives,exclusions},{IfNoneMatch:'*'});
 // Archive metadata is private to authors; teammate keys can only see viewer/*.
 const view={schema:next.schema,createdAt:next.createdAt,files:next.files};
 await putJSON(snapshot,view,{IfNoneMatch:'*'});
 await putJSON(config.catalogKey,{schema:next.schema,snapshot,archive:archiveKey,createdAt:next.createdAt},previous.etag?{IfMatch:previous.etag}:{IfNoneMatch:'*'});
 advanced=true;
 const report={snapshot,archive:archiveKey,assets:Object.keys(next.files).length,archiveFiles:Object.keys(next.archives).length,uploaded,uploadedBytes:bytes,excluded:exclusions};
 await writeFile(path.join(cacheDir,'last-publish.json'),JSON.stringify(report,null,2));
 console.log(JSON.stringify(report,null,2));
}catch(e){console.error(`Publish failed (${e.name}): ${e.message}. ${advanced ? 'S3 snapshot was published; local reporting failed.' : 'Shared latest pointer was not advanced by this attempt; retry bun run runs:publish.'}`);process.exitCode=1;}
finally{clearInterval(progress);await writeFile(cacheFile,JSON.stringify(cache));s3.destroy();}
