import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
    stages: [
        { duration: '30s', target: 50 },  // Ramp up to 50 users
        { duration: '1m', target: 50 },   // Stay at 50 users for 1 min
        { duration: '30s', target: 0 },   // Ramp down to 0 users
    ],
    thresholds: {
        http_req_duration: ['p(95)<1500'], // P95 latency should be < 1.5s
        http_req_failed: ['rate<0.01'],    // Error rate should be < 1%
    },
};

const API_BASE_URL = 'http://localhost:8000';

export default function () {
    const payload = JSON.stringify({
        age: 35,
        recency: 45,
        frequency: 10,
        monetary: 1500.50
    });

    const params = {
        headers: {
            'Content-Type': 'application/json',
            'X-API-Key': 'neural_secret_key_prod_123!' // Matches settings.py dummy key
        },
    };

    const res = http.post(`${API_BASE_URL}/predict/churn`, payload, params);
    
    check(res, {
        'status is 200': (r) => r.status === 200,
        'has churn prediction': (r) => r.json().hasOwnProperty('churn_probability'),
    });

    sleep(1);
}
