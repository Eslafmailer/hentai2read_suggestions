import {readdirSync, readFileSync} from 'fs';
import {db, reverseMapping} from "./shared";

const files = readdirSync(__dirname);
for (const file of files) {
    if (!/result.*\.json/.test(file)) {
        continue;
    }

    const result: string[] = JSON.parse(readFileSync(file, 'utf-8'));

    let unknown = 0;
    let tp = 0;
    let fp = 0;
    for (const name of result) {
        const realName = reverseMapping[name];
        if (!realName) {
            console.log(`Missing name - ${name}`);
            continue;
        }

        const book = db[realName];
        if (!book) {
            console.log(`Missing book - ${name}`);
            continue;
        }

        if (book.label == null) {
            unknown++;
        } else if (book.label) {
            tp++;
        } else {
            fp++;
        }
    }

    console.log(file, '\t', `tp: ${tp}, fp: ${fp}, unk: ${unknown}`);
}
