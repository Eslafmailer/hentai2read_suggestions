import {db, DB_FILE_NAME} from "./shared";
import {readFileSync, writeFileSync} from "fs";
import {printError} from "./utils";

const EMBEDDINGS_FILE_NAME = 'SIFT.json';
export const embeddings: Record<string, number[]> = JSON.parse(readFileSync(EMBEDDINGS_FILE_NAME, 'utf-8'));

(async () => {
    for (const [name64, embedding] of Object.entries(embeddings)) {
        const name = atob(name64);
        let book = db[name];
        if(!book) {
            throw new Error(`Unknown book - ${name}`);
        }

        book.cover = embedding;
    }

    writeFileSync(DB_FILE_NAME, JSON.stringify(db, null, 2));
})().catch(printError);