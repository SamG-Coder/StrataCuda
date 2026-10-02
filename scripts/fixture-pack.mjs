import {mkdir,writeFile} from 'node:fs/promises';
import {createFixture} from '../src/fixture.js';
import {encodeFixturePack} from '../src/pack.js';
const pack=encodeFixturePack(createFixture()),dir=new URL('../generated/fixture/',import.meta.url);
await mkdir(dir,{recursive:true});await writeFile(new URL('manifest.json',dir),JSON.stringify(pack.manifest,null,2));await writeFile(new URL('weights.bin',dir),pack.binary);
console.log(`Wrote untrained fixture pack: ${pack.binary.byteLength} bytes`);
