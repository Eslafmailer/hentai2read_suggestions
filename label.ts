import {config, db, DB_FILE_NAME, enableFavorites, loadPagedLinks, login, walkPagedLinks} from "./shared";
import {printError} from "./utils";
import {loadBook} from "./load-book";
import {writeFileSync} from "fs";
import * as process from "process";

const args = process.argv.slice(2);
const updateAll = args.some(x => x === '--update-all=true');

(async () => {
    await login();
    await enableFavorites();

    console.log('Loading True labels');
    for (const link of config.labels.true) {
        await walkPagedLinks(
            page => loadPagedLinks(page, link),
            createLabeler(true),
        );
    }

    console.log('Loading False labels');
    for (const link of config.labels.false) {
        await walkPagedLinks(
            page => loadPagedLinks(page, link),
            createLabeler(false),
        );
    }

    writeFileSync(DB_FILE_NAME, JSON.stringify(db, null, 2));
})().catch(printError);

function createLabeler(value: boolean) {
    return async function label(name: string): Promise<false | undefined> {
        let book = db[name];
        if (!book) {
            book = await loadBook(name);
            if (!book) {
                return undefined;
            }

            db[name] = book;
        }
        if(book.label === value && !updateAll) {
            return false;
        }

        book.label = value;
        return undefined;
    }
}
