const { createApp, nextTick } = Vue;

const adminApp = createApp({
    data() {
        return {
            page: document.getElementById('admin-app')?.dataset.page || 'dashboard',
            loading: false,
            overview: null,
            models: null,
            crawler: null,
            experiments: null,
            error: '',
            message: '',
            jobForm: {
                pages: 8,
                pages_per_movie: 3,
                limit_movies: 30,
                user_count: 60,
                min_behaviors: 8,
                max_behaviors: 16
            },
            experimentForm: {
                top_k: 5,
                note: ''
            },
            experimentChartMetric: 'ndcg',
            experimentBarMetric: 'ndcg_at_k',
            experimentLineChart: null,
            experimentBarChart: null
        };
    },
    computed: {
        metricRows() {
            return this.overview?.evaluation?.metrics || [];
        },
        modelVersions() {
            return this.models?.versions || this.overview?.rl?.versions || [];
        },
        crawlerRows() {
            return Object.entries(this.crawler?.crawler || this.overview?.crawler || {});
        },
        experimentRuns() {
            return this.experiments?.runs || this.overview?.experiments?.runs || [];
        },
        experimentTrends() {
            return this.experiments?.trends || this.overview?.experiments?.trends || [];
        },
        experimentLatestMetrics() {
            return this.experiments?.latest_metrics || this.overview?.experiments?.latest_metrics || [];
        }
    },
    methods: {
        async refresh() {
            this.loading = true;
            this.error = '';
            try {
                if (this.page === 'dashboard') {
                    this.overview = (await window.MovieApp.getJSON('/api/v1/admin/overview')).data;
                } else if (this.page === 'models') {
                    this.models = (await window.MovieApp.getJSON('/api/v1/admin/models')).data;
                } else if (this.page === 'crawler') {
                    this.crawler = (await window.MovieApp.getJSON('/api/v1/admin/crawler/status')).data;
                } else if (this.page === 'experiments') {
                    this.experiments = (await window.MovieApp.getJSON('/api/v1/admin/experiments')).data;
                    await nextTick();
                    this.renderExperimentCharts();
                }
            } catch (error) {
                this.error = error.message;
            } finally {
                this.loading = false;
            }
        },
        async bootstrapModel() {
            await this.runAction(() => window.MovieApp.postJSON('/api/v1/admin/models/bootstrap', {}), '已完成 bootstrap');
        },
        async trainModel(force = true) {
            await this.runAction(() => window.MovieApp.postJSON('/api/v1/admin/models/train', { force }), '已触发训练');
        },
        async rollbackModel(versionTag) {
            await this.runAction(
                () => window.MovieApp.postJSON(`/api/v1/admin/models/${versionTag}/rollback`, {}),
                `已回滚到 ${versionTag}`
            );
        },
        async runCrawler(job) {
            const payload = { job, ...this.jobForm };
            await this.runAction(
                () => window.MovieApp.postJSON('/api/v1/admin/crawler/run', payload),
                `已启动 ${job} 任务`
            );
        },
        async runExperiment() {
            await this.runAction(
                () => window.MovieApp.postJSON('/api/v1/admin/experiments/run', this.experimentForm),
                '已生成新的实验快照'
            );
        },
        async runAction(action, message) {
            this.error = '';
            this.message = '';
            this.loading = true;
            try {
                await action();
                this.message = message;
                await this.refresh();
            } catch (error) {
                this.error = error.message;
            } finally {
                this.loading = false;
            }
        },
        metricLabel(metricKey) {
            const labels = {
                ndcg: 'NDCG@K',
                recall: 'Recall@K',
                precision: 'Precision@K',
                ndcg_at_k: 'NDCG@K',
                recall_at_k: 'Recall@K',
                precision_at_k: 'Precision@K',
                coverage: 'Coverage',
                diversity: 'Diversity'
            };
            return labels[metricKey] || metricKey;
        },
        chartPalette(index) {
            const palette = [
                '#1447e6',
                '#0f766e',
                '#d97706',
                '#dc2626',
                '#7c3aed',
                '#0891b2',
                '#65a30d',
                '#c2410c',
                '#db2777',
                '#334155'
            ];
            return palette[index % palette.length];
        },
        renderExperimentCharts() {
            if (this.page !== 'experiments' || !window.Chart) {
                return;
            }
            this.renderExperimentLineChart();
            this.renderExperimentBarChart();
        },
        renderExperimentLineChart() {
            const canvas = document.getElementById('experiment-line-chart');
            if (!canvas || !this.experimentTrends.length) {
                return;
            }
            if (this.experimentLineChart) {
                this.experimentLineChart.destroy();
            }

            const datasets = this.experimentTrends.map((trend, index) => ({
                label: trend.algorithm,
                data: trend[this.experimentChartMetric] || [],
                borderColor: this.chartPalette(index),
                backgroundColor: `${this.chartPalette(index)}22`,
                borderWidth: 2,
                tension: 0.28,
                fill: false
            }));
            const labels = this.experimentTrends[0]?.labels || [];
            this.experimentLineChart = new window.Chart(canvas, {
                type: 'line',
                data: {
                    labels,
                    datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false
                    },
                    plugins: {
                        legend: {
                            position: 'bottom'
                        },
                        title: {
                            display: true,
                            text: `${this.metricLabel(this.experimentChartMetric)} 趋势`
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            suggestedMax: 1
                        }
                    }
                }
            });
        },
        renderExperimentBarChart() {
            const canvas = document.getElementById('experiment-bar-chart');
            if (!canvas || !this.experimentLatestMetrics.length) {
                return;
            }
            if (this.experimentBarChart) {
                this.experimentBarChart.destroy();
            }

            const labels = this.experimentLatestMetrics.map((metric) => metric.algorithm);
            const values = this.experimentLatestMetrics.map((metric) => metric[this.experimentBarMetric] || 0);
            const colors = this.experimentLatestMetrics.map((_, index) => this.chartPalette(index));
            this.experimentBarChart = new window.Chart(canvas, {
                type: 'bar',
                data: {
                    labels,
                    datasets: [
                        {
                            label: this.metricLabel(this.experimentBarMetric),
                            data: values,
                            backgroundColor: colors
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        },
                        title: {
                            display: true,
                            text: `最新快照 ${this.metricLabel(this.experimentBarMetric)} 对比`
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            suggestedMax: 1
                        }
                    }
                }
            });
        }
    },
    mounted() {
        this.refresh();
    }
});

adminApp.config.compilerOptions.delimiters = ['[[', ']]'];
adminApp.mount('#admin-app');
