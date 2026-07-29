(function () {
  'use strict';

  const BASE = '/api';

  window.API = {
    async openProject(path) {
      return (await fetch(BASE + '/project/open', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path }),
      })).json();
    },

    async getFrame(idx) {
      return await fetch(BASE + '/preview/frame/' + idx);
    },

    async analyzeFrame(data) {
      return (await fetch(BASE + '/preview/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })).json();
    },

    async recomposite(data) {
      return (await fetch(BASE + '/preview/recomposite', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })).json();
    },

    async getTimelineFaces() {
      return (await fetch(BASE + '/timeline/faces')).json();
    },

    async updateCut(data) {
      return (await fetch(BASE + '/timeline/cut', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })).json();
    },

    async startExport(data) {
      return (await fetch(BASE + '/export/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      })).json();
    },

    async getExportProgress(id) {
      return (await fetch(BASE + '/export/progress/' + id)).json();
    },

    async cancelExport(id) {
      return (await fetch(BASE + '/export/cancel/' + id, { method: 'POST' })).json();
    },

    async getModels() {
      return (await fetch(BASE + '/models')).json();
    },

    async loadModel(name) {
      return (await fetch(BASE + '/models/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      })).json();
    },
  };
})();
