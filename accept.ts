import {readFileSync} from "fs";
import {printError, promiseAll} from "./utils";
import {addToFavorites, db, enableFavorites, reverseMapping} from "./shared";
import {loadBook} from "./load-book";

export const RESULT_FILE_NAME = 'result.json';
export const result: string[] = JSON.parse(readFileSync(RESULT_FILE_NAME, 'utf-8'));

(async () => {
    await enableFavorites();

    await promiseAll(result, async name_ => {
        const name = reverseMapping[name_];
        if(!name) {
            console.warn(`Can't de-anonymize: ${name_}`);
            return;
        }

        const existingBook = db[name];
        if(existingBook?.label !== undefined) {
            console.warn(`Already marked: ${name}`);
            return;
        }

        const book = await loadBook(name);
        if(!book) {
            console.warn(`Book not found: ${name}`);
            return;
        }

        await addToFavorites(book.id, 4);
    });
})().catch(printError);