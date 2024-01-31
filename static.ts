import {join} from 'path';
import {Book, config, db} from "./shared";
import {printError} from "./utils";
import {writeFileSync} from "fs";

const IMAGES_FOLDER = join(__dirname, 'static');
const BOOKS_FILE_NAME = join(IMAGES_FOLDER, 'books.js');

interface BookEx extends Book {
    href: string;
    coverUrl: string;
}

(async () => {
    const data: BookEx[] = [];
    for (const book of Object.values(db)) {
        if(!book.label) {
            continue;
        }
        data.push({
            ...book,
            href: `${config.url}/${book.name}/`,
            coverUrl: `images/${book.id}.jpg`,
        });
    }
    writeFileSync(BOOKS_FILE_NAME, `var books = ${JSON.stringify(data, null, 2)};`);
})().catch(printError);