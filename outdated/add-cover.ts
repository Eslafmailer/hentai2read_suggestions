import {db, DB_FILE_NAME} from "../shared";
import {readFileSync, writeFileSync} from "fs";
import {printError} from "../utils";

const FILE_NAME_ARG = '--file=';

const args = process.argv.slice(2);
const embeddingsFileName = args.find(x => x.startsWith(FILE_NAME_ARG))?.slice(FILE_NAME_ARG.length);
if (!embeddingsFileName) {
    throw new Error(`Missing --file parameter`);
}

export const embeddings: Record<string, number[]> = JSON.parse(readFileSync(embeddingsFileName, 'utf-8'));

(async () => {
    for (const [name64, embedding] of Object.entries(embeddings)) {
        const name = atob(name64);
        let book = db[name];
        if (!book) {
            throw new Error(`Unknown book - ${name}`);
        }

        book.cover = embedding;
    }

    writeFileSync(DB_FILE_NAME, JSON.stringify(db, null, 2));
})().catch(printError);