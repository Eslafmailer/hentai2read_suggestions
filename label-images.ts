import {join} from 'path';
import {db} from "./shared";
import {existsSync, readFileSync, writeFileSync, mkdirSync} from "fs";
import {FILES_FOLDER} from "./load-book";

const IMAGES_FOLDER = join(__dirname, 'images');
const TRUE_FOLDER = join(IMAGES_FOLDER, 'true');
const FALSE_FOLDER = join(IMAGES_FOLDER, 'false');

if(!existsSync(IMAGES_FOLDER)) {
    mkdirSync(IMAGES_FOLDER);
}
if(!existsSync(TRUE_FOLDER)) {
    mkdirSync(TRUE_FOLDER);
}
if(!existsSync(FALSE_FOLDER)) {
    mkdirSync(FALSE_FOLDER);
}

for(const book of Object.values(db)) {
    if(book.label == undefined) {
        continue;
    }

    const coverFile = join(FILES_FOLDER, String(book.id));
    if(!existsSync(coverFile)) {
        continue;
    }

    const content = readFileSync(coverFile, 'utf-8');
    const buffer = Buffer.from(content, 'base64');
    writeFileSync(join(book.label ? TRUE_FOLDER : FALSE_FOLDER, String(book.id) + '.jpg'), buffer);
}