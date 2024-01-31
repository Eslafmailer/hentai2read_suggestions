import {writeFileSync} from "fs";
import {printError} from "./utils";
import {db, DB_FILE_NAME, enableFavorites, Links, loadPagedLinks, walkPagedLinks} from "./shared";
import {loadBook} from "./load-book";
import * as process from "process";
import moment from 'moment';

const PAGE_ARG = '--page=';
const UPDATE_MONTHS_ARG = '--update-months=';

const args = process.argv.slice(2);
const updateMonthsStr = args.find(x => x.startsWith(UPDATE_MONTHS_ARG))?.slice(UPDATE_MONTHS_ARG.length);
const updateMonths = updateMonthsStr ? Number(updateMonthsStr) : undefined;
const stopAfter = updateMonths ? moment().subtract(updateMonths, 'months').valueOf() : undefined;
const pageStr = args.find(x => x.startsWith(PAGE_ARG))?.slice(PAGE_ARG.length);
const page = pageStr ? Number(pageStr) : undefined;

(async () => {
    await enableFavorites();
    await walkPagedLinks(
        loadLinksFromAll,
        async name => {
            const existingBook = db[name];
            if (existingBook && (!existingBook.uploaded || !stopAfter || existingBook.uploaded < stopAfter)) {
                console.log(`Book is already in DB: ${name}`);
                return false;
            }

            const book = await loadBook(name);
            if (book) {
                db[name] = book;
            }

            return undefined;
        }, async () => {
            writeFileSync(DB_FILE_NAME, JSON.stringify(db, null, 2));
        },
        !!stopAfter,
        page);
})().catch(printError);

async function loadLinksFromAll(page: number): Promise<Links> {
    return loadPagedLinks(page, 'aGVudGFpLWxpc3QvYWxsL2FueS9hbGwvbGFzdC1hZGRlZA==');
}

