import type { Client } from '@hey-api/client-fetch';

declare module '@hey-api/client-fetch' {
  interface RequestOptions<
    MediaTypes extends string = string,
    ThrowOnError extends boolean = boolean,
    Url extends string = string,
  > {
    client?: Client;
  }
}
