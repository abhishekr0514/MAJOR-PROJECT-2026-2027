import { client } from './sdk.gen';

// Set base URL to FastAPI backend
client.setConfig({
  baseUrl: 'http://localhost:8000',
});

// Attach JWT token from localStorage on all outgoing requests
client.interceptors.request.use((request) => {
  const token = localStorage.getItem('access_token') || localStorage.getItem('token');
  if (token) {
    request.headers.set('Authorization', `Bearer ${token}`);
  }
  return request;
});
