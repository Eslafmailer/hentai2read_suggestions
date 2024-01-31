import {db, DB_FILE_NAME, enableFavorites, loadPagedLinks, walkPagedLinks} from "./shared";
import {printError} from "./utils";
import {loadBook} from "./load-book";
import {writeFileSync} from "fs";

(async () => {
    await enableFavorites();

    console.log('Loading False labels');
    await walkPagedLinks(
        loadFalseLinks,
        createLabeler(false),
    );

    console.log('Loading True labels');
    await walkPagedLinks(
        loadTrueLinks,
        createLabeler(true),
    );

    console.log('Loading True labels2');
    await walkPagedLinks(
        loadTrueLinks2,
        createLabeler(true),
    );

    writeFileSync(DB_FILE_NAME, JSON.stringify(db, null, 2));
})().catch(printError);

function createLabeler(value: boolean) {
    return async function label(name: string) {
        let book = db[name];
        if (!book) {
            book = await loadBook(name);
            if (!book) {
                return;
            }

            db[name] = book;
        }

        book.label = value;
    }
}

async function loadTrueLinks(page: number): Promise<{ names: string[], last: boolean }> {
    return loadPagedLinks(page, 'Ym9va21hcmsvcmVhZGluZy9hbGwvbmFtZS1heg==');
}
async function loadTrueLinks2(page: number): Promise<{ names: string[], last: boolean }> {
    return await loadPagedLinks(page, 'Ym9va21hcmsvcGxhbi10by1yZWFkL2FsbC9uYW1lLWF6');
}

async function loadFalseLinks(page: number): Promise<{ names: string[], last: boolean }> {
    return loadPagedLinks(page, 'Ym9va21hcmsvb24taG9sZC9hbGwvbmFtZS1heg==');
}