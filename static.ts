import {join} from 'path';
import {Book, db, URL} from "./shared";
import {printError} from "./utils";
import {writeFileSync, existsSync, readFileSync} from "fs";

const IMAGES_FOLDER = join(__dirname, 'static');
const IMAGES_SCORES_FILE_NAME = join(__dirname, 'cover-vgg.json');
const BOOKS_FILE_NAME = join(IMAGES_FOLDER, 'books.js');

const imageScores: {id: string, cover: string}[] = existsSync(IMAGES_SCORES_FILE_NAME) ? JSON.parse(readFileSync(IMAGES_SCORES_FILE_NAME, 'utf-8')) : [];
const imageScoresMap: Record<number, number> = (() => {
    const result: Record<number, number> = {};
    for (const item of imageScores) {
        result[item.id] = item.cover;
    }
    return result;
})();

interface BookEx extends Book {
    href: string;
    coverScore?: number;
}

(async () => {
    const data: BookEx[] = [];
    for (const book of Object.values(db)) {
        if(!book.label) {
            continue;
        }
        data.push({
            ...book,
            href: `${URL}/${book.name}/`,
            coverScore: imageScoresMap[book.id],
        });
    }
    writeFileSync(BOOKS_FILE_NAME, `var books = ${JSON.stringify(data, null, 2)};`);
})().catch(printError);