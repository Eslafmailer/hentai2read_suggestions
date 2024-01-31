import {printError} from "./utils";
import randomWords from "random-words";
import {Book, db} from "./shared";
import {existsSync, readFileSync, writeFileSync} from "fs";

export const MAPPING_FILE_NAME = 'mapping.json';
export const mapping: Record<string, string> = existsSync(MAPPING_FILE_NAME) ? JSON.parse(readFileSync(MAPPING_FILE_NAME, 'utf-8')) : {};
const usedRandomWords = new Set<string>(Object.values(mapping));
if (usedRandomWords.size !== Object.values(mapping).length) {
    throw new Error(`Duplicate keys in mapping detected!`);
}

export const DATA_FILE_NAME = 'data.json';
export const data: Book[] = [];

(async () => {
    for(const book of Object.values(db)) {
        book.name = anonymize(book.name);
        anonymizeAll('authors', book);
        anonymizeAll('categories', book);
        anonymizeAll('tags', book);

        data.push(book);
    }
    writeFileSync(DATA_FILE_NAME, JSON.stringify(data, null, 2));
    writeFileSync(MAPPING_FILE_NAME, JSON.stringify(mapping, null, 2));
})().catch(printError);

function anonymizeAll<K extends 'authors' | 'categories' | 'tags'>(key: K, book: Book) {
    book[key] = book[key].map(x => anonymize(x));
}
function anonymize(value: string): string {
    let anonymized = mapping[value];
    if(!anonymized) {
        anonymized = mapping[value] = generate();
    }

    return anonymized;
}

function generate(length: number = 2): string {
    while (true) {
        const words = randomWords({exactly: length, join: '-'});
        if(!usedRandomWords.has(words)) {
            usedRandomWords.add(words);
            return words;
        }
    }
}