import { client } from './sdk.gen';

// Configure the base URL for the Hey API fetch client
client.setConfig({
  baseUrl: 'http://localhost:8000',
});

// Automatically inject JWT Bearer token if present in localStorage
client.interceptors.request.use((request) => {
  const token = localStorage.getItem('access_token') || localStorage.getItem('token');
  if (token) {
    request.headers.set('Authorization', `Bearer ${token}`);
  }
  return request;
});

export { client };
