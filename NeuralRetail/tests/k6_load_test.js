import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const demandLatency = new Trend('demand_latency');
const churnLatency = new Trend('churn_latency');

export const options = {
    stages: [
        { duration: '30s', target: 50 },    // Ramp up to 50 users
        { duration: '1m', target: 100 },     // Ramp to 100 users
        { duration: '1m', target: 200 },     // Ramp to 200 users (PDF requirement)
        { duration: '2m', target: 200 },     // Sustain 200 users for 2 min
        { duration: '30s', target: 0 },      // Ramp down
    ],
    thresholds: {
        http_req_duration: ['p(95)<1500'],   // P95 latency < 1.5s (PDF Day 28)
        http_req_failed: ['rate<0.01'],      // Error rate < 1% (PDF Day 26)
        errors: ['rate<0.01'],               // Custom error rate < 1%
    },
};

const API_BASE_URL = __ENV.API_BASE_URL || 'http://localhost:8000';

// Authenticate once per VU
function getToken() {
    const loginRes = http.post(`${API_BASE_URL}/api/v1/login/access-token`, {
        username: 'admin',
        password: 'admin',
    });
    if (loginRes.status === 200) {
        return loginRes.json().access_token;
    }
    return null;
}

export function setup() {
    // Warm up and get token
    const token = getToken();
    return { token };
}

export default function (data) {
    const headers = {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${data.token}`,
    };

    // Scenario 1: Health check (lightweight)
    group('Health Check', function () {
        const res = http.get(`${API_BASE_URL}/health`);
        check(res, {
            'health: status 200': (r) => r.status === 200,
            'health: body ok': (r) => r.json().status === 'ok',
        });
        errorRate.add(res.status !== 200);
    });

    sleep(0.5);

    // Scenario 2: Demand forecast prediction
    group('Demand Forecast', function () {
        const payload = JSON.stringify({
            sku_id: `SKU-${1001 + Math.floor(Math.random() * 10)}`,
            horizon_days: 14,
            store_id: 'STORE-001',
        });
        const res = http.post(`${API_BASE_URL}/api/v1/predict/demand`, payload, { headers });
        check(res, {
            'demand: status 200': (r) => r.status === 200,
            'demand: has prediction': (r) => r.json().hasOwnProperty('predicted_demand'),
        });
        demandLatency.add(res.timings.duration);
        errorRate.add(res.status !== 200);
    });

    sleep(0.5);

    // Scenario 3: Churn prediction with SHAP
    group('Churn Prediction', function () {
        const payload = JSON.stringify({
            customer_id: `CUST-${String(Math.floor(Math.random() * 5000) + 1).padStart(4, '0')}`,
        });
        const res = http.post(`${API_BASE_URL}/api/v1/predict/churn`, payload, { headers });
        check(res, {
            'churn: status 200': (r) => r.status === 200,
            'churn: has probability': (r) => r.json().hasOwnProperty('churn_probability'),
        });
        churnLatency.add(res.timings.duration);
        errorRate.add(res.status !== 200);
    });

    sleep(0.5);

    // Scenario 4: Inventory reorder
    group('Inventory Reorder', function () {
        const payload = JSON.stringify({
            sku_id: `SKU-${1001 + Math.floor(Math.random() * 20)}`,
            store_id: 'STORE-001',
        });
        const res = http.post(`${API_BASE_URL}/api/v1/inventory/reorder`, payload, { headers });
        check(res, {
            'inventory: status 200': (r) => r.status === 200,
            'inventory: has eoq': (r) => r.json().hasOwnProperty('recommended_reorder_qty'),
        });
        errorRate.add(res.status !== 200);
    });

    sleep(0.5);

    // Scenario 5: Segmentation
    group('Customer Segmentation', function () {
        const res = http.post(`${API_BASE_URL}/api/v1/segment/score`, '{}', { headers });
        check(res, {
            'segment: status 200': (r) => r.status === 200,
            'segment: has segments': (r) => r.json().hasOwnProperty('num_segments'),
        });
        errorRate.add(res.status !== 200);
    });

    sleep(1);
}
