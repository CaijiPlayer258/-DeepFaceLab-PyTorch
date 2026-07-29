(function () {
  'use strict';

  window.Preview = {
    init: function () {
      this.canvasOriginal = document.getElementById('canvas-original');
      this.canvasDetection = document.getElementById('canvas-detection');
      this.canvasSwapped = document.getElementById('canvas-swapped');
    },

    updateOriginal: function (imageUrl) {
      var img = new Image();
      var self = this;
      img.onload = function () {
        var ctx = self.canvasOriginal.getContext('2d');
        var parent = self.canvasOriginal.parentElement;
        var w = parent.clientWidth - 20;
        var h = parent.clientHeight - 40;
        self.canvasOriginal.width = w;
        self.canvasOriginal.height = h;
        ctx.drawImage(img, 0, 0, w, h);
      };
      img.src = imageUrl;
    },

    updateDetection: function (imageUrl, faces) {
      var img = new Image();
      var self = this;
      img.onload = function () {
        var ctx = self.canvasDetection.getContext('2d');
        var parent = self.canvasDetection.parentElement;
        var w = parent.clientWidth - 20;
        var h = parent.clientHeight - 40;
        self.canvasDetection.width = w;
        self.canvasDetection.height = h;
        ctx.drawImage(img, 0, 0, w, h);

        var countEl = document.getElementById('face-count');
        if (faces && faces.length > 0) {
          faces.forEach(function (face) {
            ctx.strokeStyle = '#5b5bd6';
            ctx.lineWidth = 1.5;
            ctx.strokeRect(face.x, face.y, face.w, face.h);
          });
          countEl.textContent = faces.length + ' faces';
        } else {
          countEl.textContent = '';
        }
      };
      img.src = imageUrl;
    },

    updateSwapped: function (imageUrl) {
      var img = new Image();
      var self = this;
      img.onload = function () {
        var ctx = self.canvasSwapped.getContext('2d');
        var parent = self.canvasSwapped.parentElement;
        var w = parent.clientWidth - 20;
        var h = parent.clientHeight - 40;
        self.canvasSwapped.width = w;
        self.canvasSwapped.height = h;
        ctx.drawImage(img, 0, 0, w, h);
      };
      img.src = imageUrl;
    },
  };
})();
