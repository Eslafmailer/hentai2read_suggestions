import {readdirSync} from 'fs';
import {join, parse, basename} from 'path';
import tqdm from 'tqdm';
import sharp from 'sharp';
import {printError} from "./utils";
import joinImages from 'join-images';
import {writeFileSync} from "fs";

const INPUT_FOLDER = join(__dirname, 'images/true');
const OUTPUT_FOLDER = join(__dirname, 'static/images');
const OUTPUT_MAPPING = join(__dirname, 'static/images.js');
const CHUNK_SIZE = 10;

const mapping: Record<string, {
    image: string;
    index: number;
}> = {};

type Chunk = { name: string, buffer: Buffer }[];
(async () => {
    const files = readdirSync(INPUT_FOLDER);

    const chunk: Chunk = [];
    for(let file of tqdm(files)) {
        const input = join(INPUT_FOLDER, file);
        const buffer = await sharp(input)
            .resize(200, 200)
            .toBuffer();
        chunk.push({
            name: parse(file).name,
            buffer,
        });
        if(chunk.length === CHUNK_SIZE) {
            await writeChunk(chunk);
            chunk.length = 0;
        }
    }

    if(chunk.length) {
        await writeChunk(chunk);
    }

    writeFileSync(OUTPUT_MAPPING, `var images = ${JSON.stringify(mapping, null, 2)};`);
})().catch(printError);

let chunks = 0;
async function writeChunk(chunk: Chunk) {
    const result = await joinImages(chunk.map(x => x.buffer));
    const output = join(OUTPUT_FOLDER, `${chunks}.jpg`);
    await result.toFile(output);
    chunks++;

    const image = basename(output);
    for (let i = 0; i < chunk.length; i++){
        const item = chunk[i];
        if(!item) {
            throw new Error(`No way!`);
        }

        const {name} = item;
        mapping[name] = {
            image,
            index: i,
        };
    }
}