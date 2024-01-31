import {serializeError} from 'serialize-error';
import CircularJSON from 'circular-json';
import {PartialDeep} from 'type-fest';

export function printError(ex: unknown) {
    if (typeof ex === 'string') {
        try {
            const parsed = JSON.parse(ex);
            console.log(red(JSON.stringify(parsed, null, 2)));
        } catch {
            console.log(red(ex));
        }
    } else {
        if (assertError<{message: string}>(ex) && ex.message) {
            console.log(red(ex.message));
        }

        console.log(stringifyError(ex));
    }
}
export function stringifyError(err: unknown): string {
    if (typeof err === 'string') {
        return err;
    }

    try {
        return err ? CircularJSON.stringify(serializeError(err), null, 2) : '';
    } catch {
        return getErrorMessage(err);
    }
}
export function getErrorMessage(err: unknown): string {
    if (err instanceof Error) {
        return err.message;
    }
    if (typeof err === 'string') {
        return err;
    }
    return 'unknown error';
}


export function assertError<T extends object>(err: unknown): err is PartialDeep<T> {
    return err != null && typeof err === 'object';
}

export function red(arg: string) {
    return `\u001B[91m${arg}\u001B[0m`;
}

export function isTruthy<T>(value: T): value is NonNullable<T> {
    return !!value;
}