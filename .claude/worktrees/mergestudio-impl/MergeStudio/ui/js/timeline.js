(function () {
  'use strict';

  window.Timeline = {
    init: function () {
      this.renderLegend();
    },

    renderLegend: function () {
      var el = document.getElementById('timeline-legend');
      if (!el) return;
      el.innerHTML =
        '<span><span class="legend-swatch" style="background:#2a0c4a"></span> 多人脸</span>' +
        '<span><span class="legend-swatch" style="background:#8a6caa"></span> 少人脸</span>' +
        '<span><span class="legend-swatch" style="background:#4c1010"></span> 无人脸</span>' +
        '<span style="margin-left:auto;"><span class="legend-swatch" style="background:repeating-linear-gradient(45deg,transparent,transparent 2px,rgba(255,60,60,0.08) 2px,rgba(255,60,60,0.08) 4px);border:1px solid rgba(255,60,60,0.3)"></span> 排除段</span>';
    },

    updatePlayhead: function (frameIdx, totalFrames) {
      var pct = totalFrames > 0 ? (frameIdx / totalFrames * 100) : 0;
      var ph = document.getElementById('playhead');
      if (ph) ph.style.left = pct + '%';
    },

    renderFaceDensityLine: function (faceData) {
      var container = document.getElementById('track-faces');
      if (!container) return;
      container.innerHTML = '<div class="face-density-line"></div>';
      var line = container.querySelector('.face-density-line');
      if (!line) return;

      if (!faceData || faceData.length === 0) {
        var seg = document.createElement('div');
        seg.className = 'face-density-line__seg';
        seg.style.flex = '1';
        seg.style.background = '#0d0d0f';
        line.appendChild(seg);
        return;
      }

      faceData.forEach(function (d) {
        var seg = document.createElement('div');
        seg.className = 'face-density-line__seg';
        seg.style.width = '2px';
        if (d.face_count === 0) {
          seg.style.background = '#4c1010';
        } else if (d.face_count <= 1) {
          seg.style.background = '#8a6caa';
        } else if (d.face_count <= 2) {
          seg.style.background = '#5a3c7a';
        } else {
          seg.style.background = '#2a0c4a';
        }
        line.appendChild(seg);
      });
    },

    renderCutSegments: function (segments) {
      var container = document.getElementById('track-cut');
      if (!container) return;
      container.innerHTML = '';
      segments.forEach(function (seg) {
        var div = document.createElement('div');
        div.className = 'cut-segment';
        div.style.left = seg.startPct + '%';
        div.style.width = (seg.endPct - seg.startPct) + '%';
        container.appendChild(div);
      });
    },
  };
})();
