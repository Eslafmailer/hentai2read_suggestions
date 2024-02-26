import {readdirSync} from 'fs';
import {join} from 'path';
import tqdm from 'tqdm';
import sharp from 'sharp';
import {printError} from "./utils";

const INPUT_FOLDER = join(__dirname, 'images/true');
const OUTPUT_FOLDER = join(__dirname, 'static/images');

(async () => {
    const files = readdirSync(INPUT_FOLDER);
    for(let file of tqdm(files)) {
        const input = join(INPUT_FOLDER, file);
        const output = join(OUTPUT_FOLDER, file);
        await sharp(input)
            .resize(200)
            .toFile(output);
    }
})().catch(printError);