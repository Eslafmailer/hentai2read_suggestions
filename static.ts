import {join} from 'path';
import {Book, db, URL} from "./shared";
import {printError} from "./utils";
import {writeFileSync} from "fs";

const IMAGES_FOLDER = join(__dirname, 'static');
const BOOKS_FILE_NAME = join(IMAGES_FOLDER, 'books.js');

interface BookEx extends Book {
    href: string;
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
        });
    }
    writeFileSync(BOOKS_FILE_NAME, `var books = ${JSON.stringify(data, null, 2)};`);
})().catch(printError);