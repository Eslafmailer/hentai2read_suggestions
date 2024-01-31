import axios, {AxiosResponse} from "axios";
import {load} from "cheerio";
import {isTruthy, promiseAll} from "./utils";
import {existsSync, readFileSync} from "fs";

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
}

export const CONFIG_FILE_NAME = 'config.json';
export const config: Config = JSON.parse(readFileSync(CONFIG_FILE_NAME, 'utf-8'));
config.url = atob(config.url)

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
    if (await addToFavorites(38385, 5)) {
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
