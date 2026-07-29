(function () {
  'use strict';

  window.Params = {
    init: function () {
      this.initDetectorDropdowns();
      this.initMergeParams();
    },

    initDetectorDropdowns: function () {
      var detectors = ['YOLOv8', 'BlazeFace', 'S3FD', 'CenterFace', 'YoloV5Face'];
      var landmarkers = ['insightface-2d106det'];
      var faceTypes = ['whole_face', 'full_face', 'half_face', 'midfull_face', 'head'];

      this.fillSelect('detector-select', detectors);
      this.fillSelect('landmarker-select', landmarkers);
      this.fillSelect('face-type-select', faceTypes);
    },

    fillSelect: function (id, options) {
      var sel = document.getElementById(id);
      if (!sel) return;
      sel.innerHTML = options.map(function (o) {
        return '<option value="' + o + '">' + o + '</option>';
      }).join('');
    },

    initMergeParams: function () {
      var container = document.getElementById('merge-params');
      if (!container) return;
      container.innerHTML =
        '<label class="param-dropdown">Mask mode' +
        '  <select id="mask-mode-select">' +
        '    <option value="4">learned-prd*learned-dst</option>' +
        '    <option value="6">XSeg-prd</option>' +
        '    <option value="8">XSeg-prd*XSeg-dst</option>' +
        '    <option value="0">full</option>' +
        '    <option value="1">dst</option>' +
        '  </select>' +
        '</label>' +
        this.sliderHTML('erode', -100, 100, 0) +
        this.sliderHTML('blur', 0, 100, 4) +
        '<div style="margin-top:4px;">' +
        '  <div style="font-size:10px;color:#999;margin-bottom:3px;">Color transfer</div>' +
        '  <div class="param-btn-group" id="color-transfer-btns">' +
        ['rct', 'lct', 'mkl', 'idt', 'sot-m', 'mix-m', 'none'].map(function (c) {
          return '<button class="param-btn ' + (c === 'rct' ? 'param-btn--active' : '') + '" data-ct="' + c + '">' + c + '</button>';
        }).join('') +
        '  </div>' +
        '</div>' +
        this.sliderHTML('face_scale', -50, 50, 0) +
        this.sliderHTML('super_res', 0, 100, 0) +
        '<div style="margin-top:4px;">' +
        '  <div style="display:flex;gap:4px;margin-bottom:2px;">' +
        ['off', 'box', 'gaussian'].map(function (s) {
          return '<button class="param-btn ' + (s === 'off' ? 'param-btn--active' : '') + '" data-sharpen="' + s + '">' + s + '</button>';
        }).join('') +
        '  </div>' +
        '</div>' +
        this.sliderHTML('motion_blur', 0, 100, 0) +
        this.sliderHTML('denoise', 0, 500, 0) +
        this.sliderHTML('bicubic', 0, 100, 0);
    },

    sliderHTML: function (label, min, max, val) {
      var pct = max > min ? ((val - min) / (max - min)) * 100 : 0;
      return '<div class="param-slider" data-param="' + label + '">' +
        '  <div class="param-slider__header">' +
        '    <span class="param-slider__label">' + label + '</span>' +
        '    <span class="param-slider__value">' + val + '</span>' +
        '  </div>' +
        '  <div class="param-slider__track">' +
        '    <div class="param-slider__fill" style="width:' + pct + '%"></div>' +
        '    <div class="param-slider__thumb" style="left:' + pct + '%"></div>' +
        '  </div>' +
        '</div>';
    },
  };
})();
