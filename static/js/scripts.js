window.MovieApp = {
    async request(url, options = {}) {
        const response = await fetch(url, options);
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.msg || result.message || '请求失败');
        }
        return result;
    },
    async getJSON(url) {
        return this.request(url, {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            }
        });
    },
    async postJSON(url, payload) {
        return this.request(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json; charset=UTF-8'
            },
            body: JSON.stringify(payload || {})
        });
    },
    async postForm(url, payload) {
        const body = new URLSearchParams();
        Object.entries(payload).forEach(([key, value]) => body.append(key, value));
        return this.request(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'
            },
            body: body.toString()
        });
    }
};
