import axios, {AxiosResponse} from "axios";
import {load} from "cheerio";
import {isTruthy, promiseAll} from "./utils";
import {existsSync, readFileSync} from "fs";
import assert from "assert";
import {loadBook} from "./load-book";

export enum Categories {
    OnHold = 'on-hold',
    PlanToRead = 'plan-to-read',
    Completed = 'completed',
    Favorite = 'favorite',
    Reading = 'reading',
}
export const CategoryIDs: Record<Categories, number> = {
    [Categories.OnHold]: 1,
    [Categories.PlanToRead]: 2,
    [Categories.Completed]: 3,
    [Categories.Favorite]: 4,
    [Categories.Reading]: 5,
}

export interface Book {
    id: number;
    name: string;
    cover?: number[];
    views: number;
    year?: number;
    pages: number;
    chapters: number;
    authors: string[];
    categories: string[];
    tags: string[];
    votes: number;
    score: number;
    uploaded?: number;
    label?: boolean;
}


export interface Config {
    url: string;
    cookie: string;
    all_pages: string;
    labels: {
        true: string[];
        false: string[];
    };
    input_category: number;
}

export const CONFIG_FILE_NAME = 'config.json';
export const config: Config = JSON.parse(readFileSync(CONFIG_FILE_NAME, 'utf-8'));
(() => {
    assert(config.url, `url property is required in config`);
    assert(config.all_pages, `all_pages property is required in config`);
    assert(config.cookie, `cookie property is required in config`);
    assert(config.input_category, `input_category property is required in config`);
    assert(config.labels, `labels property is required in config`);
    assert(config.labels.true.length, `true labels are required in config`);
    assert(config.labels.false.length, `false labels are required in config`);
})();

config.url = atob(config.url);

export type DB = Record<string, Book>;
export const DB_FILE_NAME = 'db.json';
export const db: DB = existsSync(DB_FILE_NAME) ? JSON.parse(readFileSync(DB_FILE_NAME, 'utf-8')) : {};

export const MAPPING_FILE_NAME = 'mapping.json';
export const mapping: Record<string, string> = existsSync(MAPPING_FILE_NAME) ? JSON.parse(readFileSync(MAPPING_FILE_NAME, 'utf-8')) : {};
export const reverseMapping: Record<string, string> = {};
for (const [key, value] of Object.entries(mapping)) {
    reverseMapping[value] = key;
}

export interface Links {
    names: string[];
    last: boolean;
}

export async function loadPagedLinks(page: number, path: string): Promise<Links> {
    return await retry(async () => {
        console.log(`Loading page ${page}`);
        const PAGE_SIZE = 48;
        const url = config.url + `/${atob(path)}/${page}/`;
        const data = await loadWebPage(url);
        if (!data) {
            throw new Error(`Can't load page links: ${url}`);
        }

        const $ = load(data);

        const pages = $('.pagination li').get().map(x => $(x).text().trim()).filter(isTruthy).at(-1);
        if (!pages) {
            throw new Error(`Can't find pagination ${url}`);
        }
        console.log(`Loaded page ${page}/${pages}`)

        const $links = $('.overlay-button .btn:nth-child(2)');
        const links = $links.get().map(x => $(x).attr('href')).filter(isTruthy);
        const lastPage = pages === page.toString();
        if (links.length < PAGE_SIZE && !lastPage) {
            throw new Error(`Missing items on the page (${links.length} instead of ${PAGE_SIZE}) ${url}`);
        }

        return {
            names: links.map(link => link.replace(config.url + '/', '').replace(/\/$/, '')),
            last: lastPage,
        };
    });
}

export async function loadWebPage(url: string): Promise<string | undefined> {
    return retry(async () => {
        try {
            const {data}: AxiosResponse<string> = await axios.get(url, {
                headers: {
                    'Cookie': config.cookie
                },
            });
            return data;
        } catch (ex) {
            if (ex?.['response']?.['status'] === 404) {
                return undefined;
            }

            throw ex;
        }
    });
}

export async function retry<T>(action: () => Promise<T>): Promise<T> {
    const MAX_ATTEMPTS = 10;
    let attempt = 0;

    while (true) {
        try {
            return await action();
        } catch (ex) {
            attempt++;
            if (attempt > MAX_ATTEMPTS) {
                throw ex;
            }

            await delay(1.5 ** attempt);
        }
    }
}

function delay(seconds: number): Promise<void> {
    return new Promise<void>(resolve => {
        setTimeout(resolve, seconds * 1000);
    });
}

export async function walkPagedLinks(loadLinks: (page: number) => Promise<Links>, onLink: (name: string) => Promise<boolean | void>, onPage?: () => Promise<void>, breakEarlier?: boolean, page = 1) {
    while (true) {
        const {names, last} = await loadLinks(page++);
        const results = await promiseAll(names, onLink);
        if (results.length && results.every(x => x === false) && breakEarlier) {
            console.log('breaking out of walkPagedLinks')
            break;
        }

        onPage?.();
        if (last) {
            break;
        }
    }
}

export async function enableFavorites() {
    const url = config.labels.true[0];
    assert(url);

    const page = await loadPagedLinks(1, url);
    const name = page.names[0];
    assert(name);

    const book = await loadBook(name);
    assert(book);

    const categories: (keyof typeof Categories)[] = <(keyof typeof Categories)[]><unknown>Object.values(Categories);
    const category = categories.find(x => atob(url).includes(x));
    assert(category);

    if (await addToFavorites(book.id, CategoryIDs[category])) {
        console.log('Enabled favorites');
    } else {
        throw new Error(`Failed to enable favorites`);
    }
}

export async function addToFavorites(id: number, kind: number): Promise<boolean> {
    const {data}: AxiosResponse<{
        status: number;
    }> = await axios.post(config.url + '/api', {
        controller: "manga",
        action: "bookmark",
        mid: id,
        mode: kind,
    }, {
        headers: {
            'Content-Type': 'multipart/form-data',
            'Cookie': config.cookie
        },
    });
    return data.status === 1;
}
