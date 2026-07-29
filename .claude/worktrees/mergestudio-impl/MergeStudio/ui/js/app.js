(function () {
  'use strict';

  window.App = {
    state: {
      projectPath: null,
      videoPath: null,
      totalFrames: 0,
      currentFrame: 0,
      fps: 0,
      models: [],
      videos: [],
      config: {
        face_type: 'whole_face',
        mode: 'overlay',
        mask_mode: 4,
        erode_mask_modifier: 0,
        blur_mask_modifier: 0,
        motion_blur_power: 0,
        output_face_scale: 0,
        super_resolution_power: 0,
        color_transfer_mode: 1,
        image_denoise_power: 0,
        bicubic_degrade_power: 0,
        color_degrade_power: 0,
      },
      detector: 'YOLOv8',
      landmarker: 'insightface-2d106det',
      faceDatabase: [],
      currentFaces: [],
      cutSegments: [],
      zoom: 1.0,
    },

    init() {
      this.initTransportSVG();
      this.initEventListeners();
      Params.init();
      Timeline.init();
      Preview.init();
    },

    initTransportSVG() {
      var svgs = [
        '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M4 2V14" stroke="#888" stroke-width="1.2"/><path d="M12 4L7 8L12 12Z" fill="#888" stroke="#888" stroke-width="0.5"/></svg>',
        '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M4 2V14" stroke="#ccc" stroke-width="1.2"/><path d="M12 5L8 8L12 11" stroke="#ccc" stroke-width="1.8" stroke-linecap="round"/></svg>',
        '<svg width="18" height="18" viewBox="0 0 18 18" fill="none"><path d="M6 5L14 9L6 13V5Z" fill="#ccc"/></svg>',
        '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M4 5L8 8L4 11" stroke="#ccc" stroke-width="1.8" stroke-linecap="round"/><path d="M12 2V14" stroke="#ccc" stroke-width="1.2"/></svg>',
        '<svg width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M4 4L9 8L4 12Z" fill="#888" stroke="#888" stroke-width="0.5"/><path d="M12 2V14" stroke="#888" stroke-width="1.2"/></svg>',
      ];
      document.getElementById('transport-controls').innerHTML = svgs.join('');
    },

    initEventListeners() {
      var self = this;

      document.getElementById('frame-input').addEventListener('change', function (e) {
        var val = parseInt(e.target.value);
        if (!isNaN(val) && val >= 0 && val < self.state.totalFrames) {
          self.seekFrame(val);
        }
      });

      document.getElementById('transport-controls').addEventListener('click', function (e) {
        var target = e.target.closest('svg');
        if (!target) return;
        var idx = Array.from(e.currentTarget.children).indexOf(target);
        if (idx === 0) self.seekFrame(Math.max(0, self.state.currentFrame - 30));
        else if (idx === 1) self.seekFrame(self.state.currentFrame - 1);
        else if (idx === 2) self.togglePlay();
        else if (idx === 3) self.seekFrame(self.state.currentFrame + 1);
        else if (idx === 4) self.seekFrame(Math.min(self.state.totalFrames - 1, self.state.currentFrame + 30));
      });

      document.getElementById('zoom-out').addEventListener('click', function () {
        self.state.zoom = Math.max(0.25, self.state.zoom / 2);
      });
      document.getElementById('zoom-in').addEventListener('click', function () {
        self.state.zoom = Math.min(8, self.state.zoom * 2);
      });
    },

    seekFrame(idx) {
      this.state.currentFrame = idx;
      document.getElementById('frame-input').value = idx;
      Timeline.updatePlayhead(idx, this.state.totalFrames);
      this.updateTimecode(idx);
    },

    updateTimecode(idx) {
      if (this.state.fps > 0) {
        var secs = idx / this.state.fps;
        var m = Math.floor(secs / 60);
        var s = Math.floor(secs % 60);
        var cs = Math.floor((secs % 1) * 100);
        document.getElementById('timecode').textContent =
          (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s + '.' + (cs < 10 ? '0' : '') + cs;
      }
    },

    togglePlay() {
      // Play/pause stub
    },

    openProject(path) {
      var self = this;
      API.openProject(path).then(function (data) {
        if (data.status === 'ok') {
          self.state.projectPath = path;
          self.state.videoPath = data.video_path;
          self.state.totalFrames = data.total_frames;
          self.state.fps = data.fps;
          document.getElementById('workspace-path').textContent = path;
          document.getElementById('frame-total').textContent = data.total_frames;
          self.renderModelList(data.models);
          self.renderVideoList(data.videos);
          self.seekFrame(0);
        }
      });
    },

    renderModelList(models) {
      var container = document.getElementById('model-list');
      container.innerHTML = '';
      models.forEach(function (m) {
        var item = document.createElement('div');
        item.className = 'list-item' + (m.format === 'dfm' ? ' list-item--active' : ' list-item--disabled');
        item.innerHTML =
          '<span class="list-item__name">' + m.name + '</span>' +
          '<span class="list-item__meta">' + m.format + ' · ' + m.size_mb + 'MB</span>' +
          (m.format !== 'dfm' ? '<span class="list-item__action">导出</span>' : '');
        container.appendChild(item);
      });
    },

    renderVideoList(videos) {
      var container = document.getElementById('video-list');
      container.innerHTML = '';
      videos.forEach(function (v) {
        var item = document.createElement('div');
        item.className = 'list-item list-item--active';
        item.innerHTML = '<span class="list-item__name">' + v + '</span>';
        container.appendChild(item);
      });
    },
  };

  document.addEventListener('DOMContentLoaded', function () { window.App.init(); });
})();
