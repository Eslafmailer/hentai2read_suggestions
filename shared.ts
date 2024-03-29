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
    login: string;
    password: string;
    all_pages: string;
    labels: {
        true: string[];
        false: string[];
    };
    input_category: number;
}

export const URL = atob('aHR0cHM6Ly9oZW50YWkycmVhZC5jb20=');
export const CONFIG_FILE_NAME = 'config.json';
export const config: Config = JSON.parse(readFileSync(CONFIG_FILE_NAME, 'utf-8'));
(() => {
    assert(config.login, `login property is required in config`);
    assert(config.password, `password property is required in config`);
    assert(config.all_pages, `all_pages property is required in config`);
    assert(config.input_category, `input_category property is required in config`);
    assert(config.labels, `labels property is required in config`);
    assert(config.labels.true.length, `true labels are required in config`);
    assert(config.labels.false.length, `false labels are required in config`);
})();

let authCookie: string = '';

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

export async function loadPagedLinks(page: number, path: string, retryAttempts?: number): Promise<Links> {
    return await retry(async () => {
        console.log(`Loading page ${page}`);
        const PAGE_SIZE = 48;
        const url = URL + `/${atob(path)}/${page}/`;
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
            names: links.map(link => link.replace(URL + '/', '').replace(/\/$/, '')),
            last: lastPage,
        };
    }, retryAttempts);
}

export async function loadWebPage(url: string): Promise<string | undefined> {
    return retry(async () => {
        try {
            const {data}: AxiosResponse<string> = await axios.get(url, {
                headers: {
                    'Cookie': authCookie
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

export async function retry<T>(action: () => Promise<T>, attempts = 10): Promise<T> {
    let attempt = 0;

    while (true) {
        try {
            return await action();
        } catch (ex) {
            attempt++;
            if (attempt >= attempts) {
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

export async function walkPagedLinks(loadLinks: (page: number) => Promise<Links>, onLink: (name: string) => Promise<boolean | void>, onPage?: () => Promise<void>, page = 1) {
    while (true) {
        const {names, last} = await loadLinks(page++);
        const results = await promiseAll(names, onLink);
        if (results.length && results.every(x => x === false)) {
            console.log('breaking out of walkPagedLinks')
            break;
        }

        onPage?.();
        if (last) {
            break;
        }
    }
}

export async function login() {
    const response: AxiosResponse<{
        status: number;
    }> = await axios.post(URL + '/login', {
        log: atob(config.login),
        pwd: atob(config.password),
        testcookie: '1'
    }, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });

    // @ts-ignore
    const cookies: string[] = response.headers.get('set-cookie');
    const cookie = cookies.find(x => x.startsWith('wordpress_logged_in_'));
    if(!cookie) {
        throw new Error(`Failed to login, check credentials`);
    }

    const cookieValue = cookie.split(';')[0];
    assert(cookieValue, `can't extract login cookie value`);
    authCookie = cookieValue;
    console.log('Login successful');
}
export async function enableFavorites() {
    console.log('Enabling favorites');

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
    }> = await axios.post(URL + '/api', {
        controller: "manga",
        action: "bookmark",
        mid: id,
        mode: kind,
    }, {
        headers: {
            'Content-Type': 'multipart/form-data',
            'Cookie': authCookie
        },
    });
    return data.status === 1;
}
