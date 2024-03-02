import {writeFileSync} from "fs";
import {printError} from "./utils";
import {config, db, DB_FILE_NAME, enableFavorites, loadPagedLinks, login, walkPagedLinks} from "./shared";
import {loadBook} from "./load-book";
import * as process from "process";
import moment from 'moment';

const PAGE_ARG = '--page=';
const UPDATE_MONTHS_ARG = '--update-months=';

const args = process.argv.slice(2);
const updateAll = args.some(x => x === '--update-all=true');
const updateMonthsStr = args.find(x => x.startsWith(UPDATE_MONTHS_ARG))?.slice(UPDATE_MONTHS_ARG.length);
const updateMonths = updateMonthsStr ? Number(updateMonthsStr) : undefined;
const stopAfter = updateMonths ? moment().subtract(updateMonths, 'months').valueOf() : undefined;
const pageStr = args.find(x => x.startsWith(PAGE_ARG))?.slice(PAGE_ARG.length);
const page = pageStr ? Number(pageStr) : undefined;

(async () => {
    await login();
    await enableFavorites();
    console.log('Scrapping');
    await walkPagedLinks(
        page => loadPagedLinks(page, config.all_pages),
        async name => {
            const existingBook = db[name];
            if (existingBook && !updateAll && (!stopAfter || (existingBook.uploaded && existingBook.uploaded < stopAfter))) {
                return false;
            }

            const book = await loadBook(name);
            if (book) {
                book.label = existingBook?.label;
                db[name] = book;
            }

            return undefined;
        }, async () => {
            writeFileSync(DB_FILE_NAME, JSON.stringify(db, null, 2));
        },
        page);
})().catch(printError);
