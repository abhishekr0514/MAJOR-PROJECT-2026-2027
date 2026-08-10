import type { Client } from '@hey-api/client-fetch';

declare module '@hey-api/client-fetch' {
  interface RequestOptions {
    client?: Client;
  }
}
