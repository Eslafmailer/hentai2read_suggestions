import {Book, config, loadWebPage, retry} from "./shared";
import {load} from "cheerio";
import {assertError, isTruthy} from "./utils";
import moment from 'moment';
import {DurationInputArg2} from "moment/moment";
import axios from "axios";

export async function loadBook(name: string): Promise<Book | undefined> {
    console.log(`Loading book ${name}`);
    const url = config.url + `/${name}/`;
    let data: string;
    try {
        data = await loadWebPage(url);
    } catch (ex) {
        if (assertError<{ response?: {status: number} }>(ex) && ex.response?.status === 404) {
            return undefined;
        }
        throw ex;
    }
    const $ = load(data);
    const $infos = $('.list.list-simple-mini .text-primary');
    const infos = new Map<string, string[]>();
    $infos.get().map(info => {
        let $info = $(info);
        const name = $info.find('b').text().trim().toLowerCase();
        if (!name) {
            return;
        }

        const values = $info.find('a').get().map(x => $(x).text().trim().toLowerCase()).filter(isTruthy);
        infos.set(name, values);
    });

    const getViews = (): number => {
        const [value] = infos.get('view') ?? [];
        if (!value) {
            throw new Error(`Missing view section: ${url}`);
        }

        const result = /^([\d,]+) views/.exec(value);
        const viewsStr = result?.[1];
        if (!viewsStr) {
            throw new Error(`Can't parse view: '${value}' ${url}`);
        }

        let views = Number(viewsStr.replaceAll(',', ''));
        if (isNaN(views)) {
            throw new Error(`Can't parse views: '${value}' ${url}`);
        }

        return views;
    }
    const getPages = (): number => {
        const [value] = infos.get('page') ?? [];
        if (!value) {
            throw new Error(`Missing page section: ${url}`);
        }

        const result = /^([\d,]+) pages/.exec(value);
        const pagesStr = result?.[1];
        if (!pagesStr) {
            throw new Error(`Can't parse pages: '${value}' ${url}`);
        }

        let pages = Number(pagesStr.replaceAll(',', ''));
        if (isNaN(pages)) {
            throw new Error(`Can't parse pages: '${value}' ${url}`);
        }

        return pages;
    }
    const getChapters = (): number => {
        const [value] = infos.get('page') ?? [];
        if (!value) {
            throw new Error(`Missing page section: ${url}`);
        }

        const result = /([\d,]+) chapters$/.exec(value);
        if(!result) {
            return 1;
        }

        const chaptersStr = result[1];
        if (!chaptersStr) {
            throw new Error(`Can't parse chapters: '${value}' ${url}`);
        }

        let chapters = Number(chaptersStr.replaceAll(',', ''));
        if (isNaN(chapters)) {
            throw new Error(`Can't parse chapters: '${value}' ${url}`);
        }

        return chapters;
    }

    const getYear = (): number | undefined => {
        const [value] = infos.get('release year') ?? [];
        if (!value) {
            throw new Error(`Missing year section: ${url}`);
        }

        if (value === '-') {
            return undefined;
        }

        const result = /^(\d+)/.exec(value);
        if (!result) {
            throw new Error(`Can't parse year: '${value}' ${url}`);
        }

        const year = Number(result[1]);
        if (isNaN(year)) {
            throw new Error(`Can't parse year: '${value}' ${url}`);
        }

        return year;
    }

    const getAuthors = (): string[] => {
        const authors = infos.get('author');
        if (!authors) {
            throw new Error(`Missing author section: ${url}`);
        }

        return authors;
    }
    const getCategories = (): string[] => {
        const categories = infos.get('category');
        if (!categories) {
            throw new Error(`Missing category section: ${url}`);
        }

        return categories;
    }
    const getTags = (): string[] => {
        const tags = infos.get('content');
        if (!tags) {
            throw new Error(`Missing content section: ${url}`);
        }

        return tags;
    }
    const getUploaded = (): number | undefined => {
        const $chapters = $('.nav-chapters li');
        const dates = $chapters.get().map(chapter => {
            let $chapter = $(chapter);
            const uploadedText = $chapter.find('.text-muted').text().trim().toLowerCase();
            if (!uploadedText) {
                throw new Error(`Can't find uploaded ${url}`);
            }

            const regex = /uploaded .* about (\d+)(.*) ago/;
            const result = regex.exec(uploadedText);
            if(!result) {
                throw new Error(`Can't parse uploaded ${url}, ${uploadedText}`);
            }

            return moment().startOf('day').subtract(+result[1]!, <DurationInputArg2>result[2]).valueOf();
        }).filter(x => x);

        dates.sort();
        return dates[0];
    }

    const getId = (): number => {
        const midStr = $('.js-addBookmark').data('mid');
        if(!midStr) {
            throw new Error(`Can't find mid ${url}`);
        }

        const mid = Number(midStr);
        if(Number.isNaN(mid)) {
            throw new Error(`Can't parse mid ${url}: ${midStr}`);
        }

        return mid;
    }

    const getCover = async (): Promise<string | undefined> => {
        const src = $('img.img-responsive').attr('src');
        if (!src) {
            throw new Error(`image not found for ${url}`);
        }

        return retry(async () => {
            try {
                const response = await axios.get(src, {
                    responseType: 'arraybuffer'
                });
                return Buffer.from(response.data, 'binary').toString('base64');
            } catch (ex) {
                if(ex?.['response']?.['status'] === 404) {
                    return undefined;
                }

                throw ex;
            }
        });
    }

    const rating = $('.js-raty').siblings().first().text();
    const parsedRating = /score ([\d.]+)\/5 with (\d+) votes/.exec(rating);
    const [_, scoreStr, votesStr] = parsedRating ?? [];
    if (!scoreStr || !votesStr) {
        throw new Error(`Can't parse rating: '${rating}' ${url}`);
    }

    const score = Number(scoreStr);
    if (isNaN(score)) {
        throw new Error(`Can't parse score: '${rating}' ${url}`);
    }

    const votes = Number(votesStr);
    if (isNaN(votes)) {
        throw new Error(`Can't parse votes: '${rating}' ${url}`);
    }

    return {
        id: getId(),
        name,
        cover: await getCover(),
        views: getViews(),
        pages: getPages(),
        chapters: getChapters(),
        year: getYear(),
        authors: getAuthors(),
        categories: getCategories(),
        tags: getTags(),
        score,
        votes,
        uploaded: getUploaded(),
    };
}