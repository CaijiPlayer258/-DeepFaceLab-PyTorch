"""
Merger Studio — WebUI page (layout only, no backend wiring yet).
"""

HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Merger Studio</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f0f13;color:#e0e0e0;font-family:'Segoe UI',system-ui,sans-serif;height:100vh;overflow:hidden;display:flex;flex-direction:column}

/* ========== TOP BAR ========== */
#topbar{display:flex;align-items:center;gap:16px;padding:6px 16px;background:#16161e;border-bottom:1px solid #2a2a3a;min-height:52px;flex-shrink:0}
#logo{cursor:pointer;width:36px;height:36px;border-radius:6px;flex-shrink:0}
#logo:hover{opacity:.8}
#model-area{display:flex;align-items:center;gap:10px;flex:1;padding:4px 12px;background:#1a1a2e;border-radius:8px;border:1px dashed #3a3a5a;min-width:200px;max-width:360px;height:40px;font-size:12px;color:#666}
#model-area.loaded{color:#7eb8f7;border-color:#4a7ab7}
#model-area .dot{width:8px;height:8px;border-radius:50%;background:#444;flex-shrink:0}
#model-area.loaded .dot{background:#4caf50;box-shadow:0 0 6px #4caf50}
#top-spacer{flex:1}

/* ========== SIDEBAR ========== */
#sidebar{position:fixed;top:52px;left:-260px;width:260px;height:calc(100vh - 52px);background:#13131f;border-right:1px solid #2a2a3a;z-index:100;transition:left .25s ease;display:flex;flex-direction:column;padding:0;overflow-y:auto}
#sidebar.open{left:0}
#sidebar .shead{padding:16px;border-bottom:1px solid #2a2a3a;font-size:14px;font-weight:600;color:#e0e0e0}
#sidebar .snav{flex:1;padding:8px 0}
#sidebar .snav a{display:flex;align-items:center;gap:8px;padding:10px 16px;color:#aaa;text-decoration:none;font-size:13px;border-left:3px solid transparent}
#sidebar .snav a:hover{background:#1a1a2e;color:#e0e0e0;border-left-color:#4a7ab7}
#sidebar-overlay{position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:99;display:none}
#sidebar-overlay.open{display:block}

/* ========== MAIN LAYOUT ========== */
#main{display:flex;flex:1;overflow:hidden}
#left-panel{width:280px;flex-shrink:0;display:flex;flex-direction:column;background:#111118;min-width:160px}
#center-panel{flex:1;display:flex;flex-direction:column;background:#0a0a10;position:relative;min-width:200px}
#preview-area{flex:1;display:flex;align-items:center;justify-content:center;color:#333;font-size:14px;overflow:hidden}
#preview-controls{display:none;height:36px;flex-shrink:0;background:#16161e;border-top:1px solid #2a2a3a;align-items:center;justify-content:center;gap:12px;padding:0 12px}
#preview-controls button{background:#1a1a2e;border:1px solid #3a3a5a;border-radius:4px;color:#aaa;cursor:pointer;font-size:13px;padding:3px 10px}
#preview-controls button:hover{background:#2a2a3a;color:#e0e0e0}
#preview-controls button.primary{background:#2a3a5a;border-color:#4a7ab7;font-size:15px;padding:3px 14px;color:#e0e0e0}
#preview-time{font-size:12px;color:#888;min-width:90px;text-align:center;font-family:monospace}
#right-panel{width:300px;flex-shrink:0;background:#111118;overflow-y:auto;padding:16px;min-width:200px}
.resize-handle{width:5px;flex-shrink:0;background:transparent;cursor:col-resize;transition:background .15s;z-index:50}
.resize-handle:hover,.resize-handle.active{background:#4a7ab7}

/* Left: collapsible sections (no flex-grow — heights set by JS) */
#left-panel{overflow:hidden}
.lsection{display:flex;flex-direction:column;min-height:0;overflow:hidden;transition:height .35s cubic-bezier(.25,.46,.45,.94)}
.lsection-header{cursor:pointer;padding:10px 12px;font-size:12px;color:#aaa;background:#16161e;border-bottom:1px solid #2a2a3a;display:flex;align-items:center;gap:8px;flex-shrink:0;user-select:none}
.lsection-header:hover{color:#e0e0e0}
.lsection-dot{width:8px;height:8px;border-radius:50%;background:#444;flex-shrink:0;transition:background .3s}
.lsection-dot.ok{background:#4caf50;box-shadow:0 0 6px #4caf50}
.lsection-label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.lsection-body{flex:1;overflow-y:auto;min-height:0}
.lsection-hint{flex:1;display:flex;align-items:center;justify-content:center;color:#444;font-size:12px;text-align:center;padding:20px;min-height:60px}
#file-drop{flex:1;overflow-y:auto;padding:8px;min-height:0;display:flex;align-items:center;justify-content:center;color:#555;font-size:13px;transition:border-color .2s}
#file-drop.drag-over{border-color:#4a7ab7;color:#7eb8f7}
#file-drop.has-files{border-style:solid;border-color:#3a3a5a;justify-content:flex-start;align-items:stretch;flex-direction:column}
.file-thumb{display:flex;align-items:center;gap:8px;padding:6px 8px;margin:2px 0;border-radius:6px;background:#1a1a2e;cursor:pointer;font-size:11px;color:#aaa;flex-shrink:0}
.file-thumb:hover{background:#2a2a5a;color:#e0e0e0}
.file-thumb.active{background:#2a3a5a;border:1px solid #4a7ab7;color:#7eb8f7}
.file-thumb img{width:64px;height:36px;border-radius:3px;object-fit:contain;background:#000;flex-shrink:0}
.file-thumb span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* Center: video preview */
#preview-area video,#preview-area img{max-width:100%;max-height:100%;display:none}
#preview-area .placeholder{font-size:48px;opacity:.15}

/* Right: params */
.param-group{margin-bottom:20px}
.param-group h3{font-size:11px;color:#666;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;border-bottom:1px solid #2a2a3a;padding-bottom:6px}
.param-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;font-size:12px}
.param-row label{color:#aaa}
.param-row input,.param-row select{width:120px;padding:4px 8px;background:#1a1a2e;border:1px solid #3a3a5a;border-radius:4px;color:#e0e0e0;font-size:12px}
.param-row input:focus,.param-row select:focus{outline:none;border-color:#4a7ab7}
.param-row input[type=range]{width:120px;padding:0}
.btn{display:inline-block;padding:6px 16px;background:#2a2a3a;border:1px solid #3a3a5a;border-radius:6px;color:#7eb8f7;cursor:pointer;font-size:12px;text-align:center}
.btn:hover{background:#3a3a5a}
.btn.primary{background:#2a3a5a;border-color:#4a7ab7}
.btn.primary:hover{background:#3a4a6a}

/* ========== BOTTOM TIMELINE ========== */
#timeline-wrap{min-height:100px;max-height:300px;height:140px;flex-shrink:0;background:#111118;border-top:1px solid #2a2a3a;display:flex;flex-direction:column;position:relative}
#timeline-resize-handle{position:absolute;top:-3px;left:0;right:0;height:7px;cursor:ns-resize;z-index:50;background:transparent}
#timeline-resize-handle:hover{background:rgba(74,122,183,.3)}
#timeline-ruler{height:20px;flex-shrink:0;position:relative;overflow:hidden;background:#0d0d15;border-bottom:1px solid #2a2a3a}
#timeline-ruler canvas{position:absolute;top:0;left:0;height:100%}
#timeline-controls{display:flex;align-items:center;gap:8px;height:24px;padding:0 12px;font-size:10px;color:#666;flex-shrink:0}
#timeline-controls button{background:#1a1a2e;border:1px solid #3a3a5a;border-radius:3px;color:#aaa;cursor:pointer;font-size:12px;padding:2px 8px;line-height:1}
#timeline-controls button:hover{background:#2a2a3a;color:#e0e0e0}
#timeline-controls input[type=range]{width:80px;height:4px;-webkit-appearance:none;background:#2a2a3a;border-radius:2px;outline:none}
#timeline-controls input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;background:#4a7ab7;cursor:pointer}
#timeline-tracks{flex:1;position:relative;overflow:hidden;background:#0a0a10;border-radius:0 0 4px 4px}
#timeline-tracks canvas{position:absolute;top:0;left:0;height:100%}
#timeline-cursor{position:absolute;top:0;width:2px;height:100%;background:#f44;z-index:10;pointer-events:none;box-shadow:0 0 4px #f44}
#timeline-pos{font-size:11px;color:#7eb8f7;min-width:60px;text-align:right}

/* ========== SCROLLBAR ========== */
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#2a2a3a;border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:#3a3a5a}

/* Cache mgmt */
#cache-list{max-height:200px;overflow-y:auto}
.cache-item{display:flex;align-items:center;justify-content:space-between;padding:6px 8px;margin:2px 0;background:#1a1a2e;border-radius:4px;font-size:11px}
.cache-item .info{flex:1;overflow:hidden}
.cache-item .name{color:#aaa;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cache-item .detail{color:#555;font-size:10px}
.cache-item .del{color:#f88;cursor:pointer;padding:2px 6px;border-radius:3px;font-size:12px}
.cache-item .del:hover{background:#5a2a2a}
.cache-empty{color:#444;font-size:12px;text-align:center;padding:20px}
.cache-total{color:#666;font-size:11px;text-align:right;padding:4px 0}
/* Progress overlay */
#progress-overlay{position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:1000;display:none;align-items:center;justify-content:center}
#progress-panel{background:#1a1a2e;border:1px solid #3a3a5a;border-radius:12px;padding:24px 32px;min-width:420px;text-align:center}
#progress-panel h3{color:#e0e0e0;margin-bottom:16px;font-size:16px}
.stage-row{display:flex;align-items:center;gap:10px;margin-bottom:10px;font-size:12px}
.stage-label{width:80px;color:#aaa;text-align:right;flex-shrink:0}
.stage-bar-bg{flex:1;height:8px;background:#2a2a3a;border-radius:4px;overflow:hidden}
.stage-bar-fill{height:100%;width:0%;border-radius:4px;transition:width .3s}
.stage-pct{width:40px;color:#888;text-align:right;font-family:monospace;font-size:11px}
.stage-done .stage-bar-fill{background:#4caf50}
.stage-active .stage-bar-fill{background:#4a7ab7}
.stage-pending .stage-bar-fill{background:#333}
.stage-label.active{color:#7eb8f7;font-weight:600}
.stage-label.done{color:#4caf50}
#progress-time{color:#666;font-size:11px;margin:12px 0 16px}
#progress-cancel{background:#5a2a2a;border:1px solid #8a3a3a;border-radius:6px;color:#f88;cursor:pointer;padding:8px 24px;font-size:13px}
#progress-cancel:hover{background:#6a3a3a}
#preview-area{flex-direction:row;gap:4px;padding:4px}
.preview-panel{flex:1;position:relative;background:#0a0a10;border-radius:4px;overflow:hidden;display:flex;align-items:center;justify-content:center;min-height:100px;min-width:80px}
.preview-panel canvas{max-width:100%;max-height:100%;object-fit:contain}
.preview-label{position:absolute;top:4px;left:6px;font-size:10px;color:#888;background:rgba(0,0,0,.6);padding:2px 6px;border-radius:3px;z-index:5;pointer-events:none}
.preview-loading{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.6);color:#aaa;font-size:13px;z-index:10;pointer-events:none}
/* Export dialog */
.export-dialog-overlay{position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:1000;display:none;align-items:center;justify-content:center}
.export-dialog{background:#1a1a2e;border:1px solid #3a3a5a;border-radius:12px;padding:24px 32px;min-width:400px;max-width:500px}
.export-dialog h3{color:#e0e0e0;margin-bottom:16px;font-size:16px}
.export-dialog .row{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;font-size:13px}
.export-dialog .row label{color:#aaa}
.export-dialog .row span{color:#e0e0e0;font-family:monospace}
.export-dialog .row select,.export-dialog .row input{width:140px;padding:4px 8px;background:#111118;border:1px solid #3a3a5a;border-radius:4px;color:#e0e0e0;font-size:12px}
.export-dialog .row select:focus,.export-dialog .row input:focus{outline:none;border-color:#4a7ab7}
.export-dialog .row input[disabled]{opacity:.5;cursor:not-allowed}
.export-dialog .btns{display:flex;gap:10px;justify-content:flex-end;margin-top:20px}
.export-dialog .btns button{padding:8px 24px;border-radius:6px;cursor:pointer;font-size:13px}
.export-dialog .btn-cancel{background:#2a2a3a;border:1px solid #3a3a5a;color:#aaa}
.export-dialog .btn-cancel:hover{background:#3a3a5a}
.export-dialog .btn-confirm{background:#2a3a5a;border:1px solid #4a7ab7;color:#7eb8f7}
.export-dialog .btn-confirm:hover{background:#3a4a6a}
/* Waiting UI (Step 2.5) */
.waiting-ui-overlay{position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:1000;display:none;align-items:center;justify-content:center}
.waiting-ui{background:#1a1a2e;border:1px solid #3a3a5a;border-radius:12px;padding:24px 32px;min-width:420px;text-align:center}
.waiting-ui h3{color:#e0e0e0;margin-bottom:16px;font-size:16px}
.waiting-ui p{color:#aaa;font-size:13px;margin-bottom:12px}
.waiting-ui .dir-path{background:#111118;border:1px solid #2a2a3a;border-radius:6px;padding:8px 12px;margin-bottom:16px;font-size:12px;color:#888;word-break:break-all;text-align:left}
.waiting-ui .dir-path span{color:#7eb8f7}
.waiting-ui .btns{display:flex;flex-wrap:wrap;gap:8px;justify-content:center}
.waiting-ui .btns button{padding:8px 20px;border-radius:6px;cursor:pointer;font-size:13px;min-width:120px}
.waiting-ui .btn-open{background:#2a3a5a;border:1px solid #4a7ab7;color:#7eb8f7}
.waiting-ui .btn-open:hover{background:#3a4a6a}
.waiting-ui .btn-cluster{background:#2a5a3a;border:1px solid #4a8a6a;color:#7fc89f}
.waiting-ui .btn-cluster:hover{background:#3a6a4a}
.waiting-ui .btn-organize{background:#5a4a2a;border:1px solid #8a7a4a;color:#d4c07f}
.waiting-ui .btn-organize:hover{background:#6a5a3a}
.waiting-ui .btn-continue{background:#2a5a4a;border:1px solid #4a8a7a;color:#7fd4bf}
.waiting-ui .btn-continue:hover{background:#3a6a5a}
.waiting-ui .btn-cancel-wait{background:#5a2a2a;border:1px solid #8a3a3a;color:#f88}
.waiting-ui .btn-cancel-wait:hover{background:#6a3a3a}
</style>
</head>
<body>

<!-- TOP BAR -->
<div id="topbar">
  <img id="logo" src="/merger-studio/logo" alt="logo" title="点击打开导航">
  <div id="top-spacer" style="flex:1"></div>
  <span style="font-size:12px;color:#555">Merger Studio</span>
</div>

<!-- SIDEBAR -->
<div id="sidebar-overlay"></div>
<div id="sidebar">
  <div class="shead">导航</div>
  <div class="snav">
    <a href="/Trainer">&#9707; 训练监控</a>
    <a href="/FlitGame">&#9673; 样本筛选</a>
    <a href="/MergerStudio" class="active">&#9678; Merger Studio</a>
    <a href="/Settings">&#9881; 参数调整</a>
  </div>
</div>

<!-- MAIN -->
<div id="main">
  <!-- LEFT: models + video sections -->
  <div id="left-panel">
    <!-- Models Section -->
    <div class="lsection" id="lsection-models">
      <div class="lsection-header" onclick="toggleSection('models')">
        <span class="lsection-dot" id="dot-models"></span>
        <span class="lsection-label" id="label-models">models</span>
      </div>
      <div class="lsection-body" id="body-models">
        <div class="lsection-hint">拖放 DFM 模型文件<br><small style="color:#444">.dfm 格式</small></div>
      </div>
    </div>
    <!-- Videos Section -->
    <div class="lsection" id="lsection-videos">
      <div class="lsection-header" onclick="toggleSection('videos')">
        <span class="lsection-dot" id="dot-videos"></span>
        <span class="lsection-label" id="label-videos">videos</span>
      </div>
      <div class="lsection-body" id="body-videos">
        <div id="file-drop" ondrop="handleDrop(event)" ondragover="handleDragOver(event)" ondragleave="handleDragLeave(event)">
          <span class="placeholder">拖放视频文件到此处<br><small style="color:#444">或点击选择文件</small></span>
          <input type="file" id="file-input" accept="video/*" multiple style="display:none" onchange="handleFiles(this.files)">
        </div>
      </div>
    </div>
  </div>
  <div class="resize-handle" id="resize-lc" data-target="left"></div>

  <!-- CENTER: video preview -->
  <div id="center-panel">
    <div id="preview-area">
      <span id="preview-placeholder" class="placeholder" style="display:none">&#9654;</span>
      <div class="preview-panel" id="panel-video">
        <div class="preview-label">视频</div>
        <video id="preview-video" style="max-width:100%;max-height:100%;border-radius:4px;object-fit:contain" onloadedmetadata="onVideoLoaded()" ontimeupdate="onVideoTimeUpdate()" onplay="onVideoPlay()" onpause="onVideoPause()" onseeked="drawTimeline()">
          <source src="" type="video/mp4">
        </video>
      </div>
      <img id="preview-img" style="display:none" alt="preview">
      <div class="preview-panel" id="panel-detection">
        <div class="preview-label">人脸检测</div>
        <canvas id="preview-detection"></canvas>
        <div class="preview-loading" id="detection-loading" style="display:none">分析中...</div>
      </div>
      <div class="preview-panel" id="panel-swapped">
        <div class="preview-label">换脸结果</div>
        <canvas id="preview-swapped"></canvas>
        <div class="preview-loading" id="swap-loading" style="display:none">合成中...</div>
      </div>
    </div>
    <div id="preview-controls">
      <button id="btn-prev-frame" title="上一帧 ←">&#9664; 帧</button>
      <button id="btn-play-pause" class="primary" title="播放/暂停 空格">&#9654;</button>
      <button id="btn-next-frame" title="下一帧 →">帧 &#9658;</button>
      <span id="preview-time">00:00:00:00</span>
    </div>
  </div>
  <div class="resize-handle" id="resize-cr" data-target="right"></div>

  <!-- RIGHT: params -->
  <div id="right-panel">
    <div class="param-group">
      <h3>&#9881; 设备与预览</h3>
      <div class="param-row"><label>GPU 设备</label><input type="text" id="cfg-device" value="0" style="width:80px" title="GPU索引或cpu"></div>
      <div class="param-row"><label>预览缩放</label><select id="cfg-preview-scale"><option value="1">1x</option><option value="2" selected>1/2</option><option value="4">1/4</option></select></div>
    </div>
    <div class="param-group">
      <h3>&#128066; 人脸提取</h3>
      <div class="param-row"><label>检测算法</label><select id="cfg-detector"><option value="S3FD">S3FD</option><option value="BlazeFace">BlazeFace</option><option value="CenterFace">CenterFace</option><option value="YoloV5Face">YoloV5Face</option><option value="YoloV8Face">YoloV8Face</option></select></div>
      <div class="param-row"><label>特征点标记</label><select id="cfg-landmark" disabled><option>insightface106pt2d (唯一)</option></select></div>
      <div class="param-row"><label>最大人脸</label><select id="cfg-max-faces"><option value="0">0 (不限)</option><option value="1" selected>1</option><option value="2">2</option><option value="3">3</option></select></div>
      <div class="param-row"><label>人脸类型</label><select id="cfg-face-type"><option value="half_face">half_face</option><option value="mid_full">mid_full</option><option value="full_face" selected>full_face</option><option value="whole_face">whole_face</option><option value="head">head</option></select></div>
    </div>
    <div class="param-group">
      <h3>&#127912; 合成模式</h3>
      <div class="param-row"><label>模式</label><select id="cfg-mode"><option value="overlay">overlay</option><option value="hist-match">hist-match</option><option value="seamless">seamless</option><option value="seamless-hist-match">seamless-hist-match</option><option value="raw-rgb">raw-rgb</option><option value="raw-predict">raw-predict</option></select></div>
      <div class="param-row"><label>直方图匹配遮罩</label><input type="checkbox" id="cfg-masked-hist-match" checked style="width:auto"></div>
    </div>
    <div class="param-group">
      <h3>&#128208; 遮罩</h3>
      <div class="param-row"><label>遮罩模式</label><select id="cfg-mask-mode"><option value="0">full</option><option value="1">dst</option><option value="2">learned-prd</option><option value="3">learned-dst</option><option value="2" selected>learned-prd</option><option value="3">learned-dst</option><option value="4">learned-prd*learned-dst</option><option value="5">learned-prd+learned-dst</option><option value="6">XSeg-prd</option><option value="7">XSeg-dst</option><option value="8">XSeg-prd*XSeg-dst</option><option value="9">learned-prd*learned-dst*XSeg-prd*XSeg-dst</option></select></div>
      <div class="param-row"><label>侵蚀</label><input type="number" id="cfg-erode-mask" value="0" min="-400" max="400" style="width:70px"></div>
      <div class="param-row"><label>模糊</label><input type="number" id="cfg-blur-mask" value="0" min="0" max="400" style="width:70px"></div>
      <div class="param-row"><label>运动模糊</label><input type="number" id="cfg-motion-blur" value="0" min="0" max="100" style="width:70px"></div>
      <div class="param-row"><label>色彩迁移</label><select id="cfg-color-transfer"><option value="None">None</option><option value="rct" selected>rct</option><option value="lct">lct</option><option value="mkl">mkl</option><option value="mkl-m">mkl-m</option><option value="idt">idt</option><option value="idt-m">idt-m</option><option value="sot-m">sot-m</option><option value="mix-m">mix-m</option></select></div>
    </div>
    <div class="param-group">
      <h3>&#9881; 后处理</h3>
      <div class="param-row"><label>人脸缩放</label><input type="number" id="cfg-face-scale" value="0" min="-50" max="50" style="width:70px"></div>
      <div class="param-row"><label>超分辨率</label><input type="number" id="cfg-super-res" value="0" min="0" max="100" style="width:70px"></div>
      <div class="param-row"><label>去噪</label><input type="number" id="cfg-denoise" value="0" min="0" max="500" style="width:70px"></div>
      <div class="param-row"><label>双三次降质</label><input type="number" id="cfg-bicubic" value="0" min="0" max="100" style="width:70px"></div>
      <div class="param-row"><label>颜色降质</label><input type="number" id="cfg-color-degrade" value="0" min="0" max="100" style="width:70px"></div>
      <div class="param-row"><label>锐化模式</label><select id="cfg-sharpen-mode"><option value="0">无</option><option value="1">方框</option><option value="2">高斯</option></select></div>
      <div class="param-row"><label>锐化强度</label><input type="number" id="cfg-sharpen-amount" value="0" min="-100" max="100" style="width:70px"></div>
    </div>
    <div class="param-group">
      <h3>&#9654; 操作</h3>
      <button class="btn" style="width:100%;margin-bottom:8px" onclick="onRenderCache(event)">&#128190; 渲染帧缓存</button>
      <button class="btn" style="width:100%;margin-bottom:8px" onclick="onExportFrame(event)">&#128196; 导出当前帧</button>
      <button class="btn primary" style="width:100%" onclick="onBatchExport()">&#9654; 开始批量导出</button>
    </div>
    <div class="param-group">
      <h3>&#128451; 缓存管理</h3>
      <div id="cache-list"><div class="cache-empty">加载中...</div></div>
    </div>
  </div>
</div>

<!-- TIMELINE -->
<div id="timeline-wrap">
  <div id="timeline-resize-handle" title="拖拽调整时间轴高度"></div>
  <div id="timeline-controls">
    <span style="color:#888;font-weight:600">&#9632; 时间轴</span>
    <button id="btn-zoom-out" title="缩小">&#8722;</button>
    <input type="range" min="1" max="500" value="100" id="zoom-slider">
    <button id="btn-zoom-in" title="放大">+</button>
    <span style="flex:1"></span>
    <span id="timeline-pos">00:00.000</span>
  </div>
  <div id="timeline-ruler"><canvas id="ruler-canvas"></canvas></div>
  <div id="timeline-tracks" onwheel="handleTimelineWheel(event)" onmousedown="handleTimelineMouseDown(event)">
    <canvas id="timeline-canvas"></canvas>
    <div id="timeline-cursor" style="left:50%"></div>
  </div>
</div>

<script>
// ========== Sidebar ==========
const sidebar=document.getElementById('sidebar');
const overlay=document.getElementById('sidebar-overlay');
document.getElementById('logo').addEventListener('click',()=>{
  sidebar.classList.toggle('open');overlay.classList.toggle('open');
});
overlay.addEventListener('click',()=>{
  sidebar.classList.remove('open');overlay.classList.remove('open');
});

// ========== Section toggle (models / videos, CSS animated) ==========
let _activeSec=null,_modelInfo=null,_videoInfo=null,_secTimer=null;

function _getHdrH(){
  let mHdr=document.getElementById('lsection-models').querySelector('.lsection-header').offsetHeight;
  let vHdr=document.getElementById('lsection-videos').querySelector('.lsection-header').offsetHeight;
  return{m:mHdr,v:vHdr};
}

function _setHeights(mH,vH){
  document.getElementById('lsection-models').style.height=mH+'px';
  document.getElementById('lsection-videos').style.height=vH+'px';
}

function toggleSection(name){
  if(_activeSec===name)return;
  _activeSec=name;
  if(_secTimer)clearTimeout(_secTimer);

  // Update labels
  if(name!=='models'&&_modelInfo)document.getElementById('label-models').textContent='models - '+_modelInfo;
  else if(!_modelInfo)document.getElementById('label-models').textContent='models';
  if(name!=='videos'&&_videoInfo)document.getElementById('label-videos').textContent='videos - '+_videoInfo;
  else if(!_videoInfo)document.getElementById('label-videos').textContent='videos';

  let panel=document.getElementById('left-panel');
  let h=_getHdrH();
  let full=Math.max(0,panel.clientHeight-h.m-h.v);

  // Step 1: collapse current expanded section to header-only
  if(name==='models'){
    _setHeights(h.m, h.v); // videos collapses to header height
    _secTimer=setTimeout(function(){
      _setHeights(h.m+full, h.v); // models expands to fill
    },360);
  }else{
    _setHeights(h.m, h.v); // models collapses to header height
    _secTimer=setTimeout(function(){
      _setHeights(h.m, h.v+full); // videos expands to fill
    },360);
  }
}
// Init: set heights instantly (no transition), then enable transition
function _initSections(){
  let mEl=document.getElementById('lsection-models'),vEl=document.getElementById('lsection-videos');
  if(!mEl||!vEl){setTimeout(_initSections,50);return;}
  mEl.style.transition='none';vEl.style.transition='none';
  _activeSec='models';
  let h=_getHdrH();
  let panel=document.getElementById('left-panel');
  let full=Math.max(0,panel.clientHeight-h.m-h.v);
  _setHeights(h.m+full, h.v);
  // Re-enable transition after a frame
  requestAnimationFrame(function(){
    requestAnimationFrame(function(){
      mEl.style.transition='';vEl.style.transition='';
    });
  });
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',_initSections);
else _initSections();
var _resizeTimer=null;
window.addEventListener('resize',function(){
  if(_resizeTimer)clearTimeout(_resizeTimer);
  let mEl=document.getElementById('lsection-models'),vEl=document.getElementById('lsection-videos');
  if(mEl)mEl.style.transition='none';
  if(vEl)vEl.style.transition='none';
  _resizeTimer=setTimeout(function(){
    if(_activeSec){
      let h=_getHdrH();
      let panel=document.getElementById('left-panel');
      let full=Math.max(0,panel.clientHeight-h.m-h.v);
      if(_activeSec==='models')_setHeights(h.m+full,h.v);
      else _setHeights(h.m,h.v+full);
    }
    if(mEl)mEl.style.transition='';
    if(vEl)vEl.style.transition='';
    _resizeTimer=null;
  },50);
});

// Auto-load .dfm models from workspace/model/
fetch('/studio/models').then(r=>r.json()).then(list=>{
  let body=document.getElementById('body-models');
  body.innerHTML='';
  if(!list||!list.length){
    body.innerHTML='<div class="lsection-hint">workspace/model/<br><small style="color:#444">未找到 .dfm 文件</small></div>';
    return;
  }
  list.forEach(m=>{
    let div=document.createElement('div');
    div.className='file-thumb';
    div.innerHTML='<span style="font-size:14px;margin-right:4px">&#128190;</span><span>'+m.name+'</span>';
    div.title=m.name+' ('+(m.size/1024/1024).toFixed(1)+' MB)';
    div.addEventListener('click',()=>{
      document.querySelectorAll('#body-models .file-thumb').forEach(t=>t.classList.remove('active'));
      div.classList.add('active');
      _onModelSelected(m.name);
      fetch('/studio/load-model', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name:m.name})})
        .then(r=>r.json()).then(d=>{
          if(d.error) console.warn('Model load failed:', d.error);
          else console.log('Model loaded:', d.path);
        }).catch(function(e){console.warn('Model load request failed',e);});
    });
    body.appendChild(div);
  });
});

function _onModelSelected(filename){
  _modelInfo=filename;
  document.getElementById('dot-models').classList.add('ok');
  document.getElementById('label-models').textContent='models - '+filename;
}
function _onVideoSelected(filename){
  _videoInfo=filename;
  document.getElementById('dot-videos').classList.add('ok');
  document.getElementById('label-videos').textContent='videos - '+filename;
}

// ========== Video preview (HTML5 video element + custom controls) ==========
const videoEl = document.getElementById('preview-video');
const previewImg = document.getElementById('preview-img');
const placeholder = document.getElementById('preview-placeholder');
const btnPP = document.getElementById('btn-play-pause');
let _currentVideoFile = null;
let _studioSid = null;

function _fmtTime(t){
  if(!t||t<0)t=0;
  const hh=Math.floor(t/3600),mm=Math.floor((t%3600)/60),ss=Math.floor(t%60),ff=Math.floor((t%1)*30);
  return String(hh).padStart(2,'0')+':'+String(mm).padStart(2,'0')+':'+String(ss).padStart(2,'0')+':'+String(ff).padStart(2,'0');
}

function onVideoPlay(){btnPP.innerHTML='&#9646;&#9646;';}
function onVideoPause(){btnPP.innerHTML='&#9654;';}

function loadVideo(file) {
  if(_currentVideoFile&&_currentVideoFile!==file){
    let oldSrc=videoEl.querySelector('source').src;
    if(oldSrc&&oldSrc.startsWith('blob:'))URL.revokeObjectURL(oldSrc);
  }
  _currentVideoFile = file;
  _onVideoSelected(file.name);
  const url = URL.createObjectURL(file);
  videoEl.querySelector('source').src = url;
  videoEl.load();
  videoEl.style.display = 'block';
  previewImg.style.display = 'none';
  placeholder.style.display = 'none';
  document.getElementById('preview-controls').style.display = 'flex';
  thumbCache=null;_pregenStrip=null;_thumbH=0;
  // Upload to ffmpeg backend for thumbnails
  fetch('/studio/upload',{method:'POST',headers:{'X-Filename':encodeURIComponent(file.name)},body:file}).then(r=>r.json()).then(d=>{
    if(d.sid){_studioSid=d.sid;thumbCache=null;_bgDirty=true;drawTimeline();}
  });
}

// Click preview to play/pause
videoEl.addEventListener('click',function(){
  if(videoEl.paused)videoEl.play();else videoEl.pause();
});

btnPP.addEventListener('click',function(){
  if(videoEl.paused)videoEl.play();else videoEl.pause();
});
document.getElementById('btn-prev-frame').addEventListener('click',function(){
  if(!videoEl.duration)return;
  if(!videoEl.paused)videoEl.pause();
  videoEl.currentTime=Math.max(0,videoEl.currentTime-1/30);
  onVideoTimeUpdate();drawTimeline();
});
document.getElementById('btn-next-frame').addEventListener('click',function(){
  if(!videoEl.duration)return;
  if(!videoEl.paused)videoEl.pause();
  videoEl.currentTime=Math.min(videoEl.duration,videoEl.currentTime+1/30);
  onVideoTimeUpdate();drawTimeline();
});

function onVideoLoaded() {
  if(videoEl.duration){
    clips[0].startSec=0;
    clips[0].endSec=videoEl.duration;
    if(inPoint===null)inPoint=0;
    if(outPoint===null)outPoint=1;
    videoEl.currentTime=Math.min(1,videoEl.duration);
  }
  _bgDirty=true;drawTimeline();
}

function syncTimelineFromVideo() {
  if (!videoEl.duration) return;
  const t = videoEl.currentTime;
  const hh = Math.floor(t / 3600);
  const mm = Math.floor((t % 3600) / 60);
  const ss = Math.floor(t % 60);
  const ff = Math.floor((t % 1) * 30);
  const ts = String(hh).padStart(2,'0')+':'+String(mm).padStart(2,'0')+':'+String(ss).padStart(2,'0')+':'+String(ff).padStart(2,'0');
  document.getElementById('timeline-pos').textContent = ts;
  document.getElementById('preview-time').textContent = ts;
}

function _setCurrentTime(sec){
  sec=Math.max(0,Math.min(videoDuration(),sec));
  videoEl.currentTime=sec;
  onVideoTimeUpdate();
}
function seekVideo(pct) {
  _setCurrentTime(pct*videoDuration());
}

// ========== File drop ==========
const dropZone=document.getElementById('file-drop');
const _addedFiles=new Set();
function handleDragOver(e){e.preventDefault();dropZone.classList.add('drag-over')}
function handleDragLeave(e){dropZone.classList.remove('drag-over')}
function handleDrop(e){
  e.preventDefault();dropZone.classList.remove('drag-over');
  handleFiles(e.dataTransfer.files);
}
function handleFiles(files){
  if(!files.length)return;
  dropZone.classList.add('has-files');
  dropZone.querySelector('.placeholder').style.display='none';
  for(let f of files){
    if(!f.type.startsWith('video/'))continue;
    let key=f.name+'|'+f.size;
    if(_addedFiles.has(key))continue;
    _addedFiles.add(key);
    let div=document.createElement('div');
    div.className='file-thumb';
    let thumb=document.createElement('img');
    div.appendChild(thumb);
    let label=document.createElement('span');
    label.textContent=f.name.substring(0,28)+(f.name.length>28?'...':'');
    div.appendChild(label);
    // 16:9 thumbnail
    let v=document.createElement('video');
    v.preload='metadata';v.muted=true;
    v.onloadeddata=()=>{v.currentTime=1};
    v.onseeked=()=>{
      let c=document.createElement('canvas');
      c.width=160;c.height=90;
      let ctx=c.getContext('2d');
      let vw=v.videoWidth,vh=v.videoHeight;
      if(!vw||!vh){vw=160;vh=90;}
      let scale=Math.min(160/vw,90/vh);
      let dw=vw*scale,dh=vh*scale;
      let dx=(160-dw)/2,dy=(90-dh)/2;
      ctx.fillStyle='rgba(0,0,0,0)';
      ctx.clearRect(0,0,160,90);
      ctx.drawImage(v,dx,dy,dw,dh);
      thumb.src=c.toDataURL();
      URL.revokeObjectURL(v.src);
    };
    v.src=URL.createObjectURL(f);

    div.addEventListener('click',(e)=>{
      e.stopPropagation();
      document.querySelectorAll('.file-thumb').forEach(t=>t.classList.remove('active'));
      div.classList.add('active');
      loadVideo(f);
    });
    dropZone.appendChild(div);
    if(!_currentVideoFile) loadVideo(f);
  }
}
dropZone.addEventListener('click',(e)=>{
  if(e.target===dropZone||e.target.classList.contains('placeholder')){
    document.getElementById('file-input').click();
  }
});
document.getElementById('file-input').addEventListener('change',function(){handleFiles(this.files);this.value='';});

// ========== Timeline (Premiere Pro style) ==========
const twrap=document.getElementById('timeline-tracks');
const tcanvas=document.getElementById('timeline-canvas');
const rcanvas=document.getElementById('ruler-canvas');
const tcursor=document.getElementById('timeline-cursor');
const tctx=tcanvas.getContext('2d');
const rctx=rcanvas.getContext('2d');
twrap.addEventListener('mousemove',function(e){
  let r=twrap.getBoundingClientRect();
  mouseTX=e.clientX-r.left;mouseTY=e.clientY-r.top;
  // Cursor feedback for draggable handles
  if(videoEl.duration){
    let x=mouseTX;
    if(inPoint!==null&&Math.abs(x-secToX(inPoint*videoDuration()))<14)twrap.style.cursor='col-resize';
    else if(outPoint!==null&&Math.abs(x-secToX(outPoint*videoDuration()))<14)twrap.style.cursor='col-resize';
    else if(Math.abs(x-secToX(videoEl.currentTime))<8)twrap.style.cursor='grab';
    else twrap.style.cursor='default';
  }else{twrap.style.cursor='default';}
});
twrap.addEventListener('mouseleave',function(){mouseTX=-100;mouseTY=-100;twrap.style.cursor='default';});

// ---- Data model ----
let clips=[{startSec:0,endSec:10,label:'Video',color:'#4a7ab7',track:0}];
let selectedClipIdx=0;
let inPoint=null,outPoint=null;
let markers=[];
let zoomLevel=100,scrollX=0;
let dragging=null,dragStartX=0,dragStartVal=0;
let _wasDrag=false;
let mouseTX=0,mouseTY=0;
let thumbCache=null;
let _bgCache=null,_bgDirty=true;
let snapThresholdPx=6;
let fps=30;

function videoDuration(){return videoEl.duration||10}
function totalFrames(){
  let vf=Math.round(videoDuration()*fps);
  let viewW=twrap.clientWidth-32;
  if(viewW<=0)return vf;
  let minF=Math.ceil(viewW*100/zoomLevel*3);
  return Math.max(vf,minF);
}
function totalDuration(){return totalFrames()/fps}
function totalWidth(){return totalFrames()*zoomLevel/100}
function clampScroll(){scrollX=Math.max(0,Math.min(scrollX,totalWidth()+32-twrap.clientWidth))}
function frameToX(f){return -scrollX+32+f*totalWidth()/totalFrames()}
function secToX(s){return frameToX(s*fps)}
function xToSec(x){return Math.max(0,(x+scrollX-32)*totalDuration()/totalWidth())}
function xToFrame(x){return Math.max(0,(x+scrollX-32)*totalFrames()/totalWidth())}


// ---- Snap system (all values in seconds) ----
function getSnapPoints(){
  let pts=[];
  if(videoEl.duration)pts.push({val:videoEl.currentTime,type:'playhead',pri:0});
  clips.forEach((c,i)=>{
    pts.push({val:c.startSec,type:'clip',pri:1,idx:i,edge:'start'});
    pts.push({val:c.endSec,type:'clip',pri:1,idx:i,edge:'end'});
  });
  if(inPoint!==null)pts.push({val:inPoint*videoDuration(),type:'in',pri:0});
  if(outPoint!==null)pts.push({val:outPoint*videoDuration(),type:'out',pri:0});
  markers.forEach((m,i)=>pts.push({val:m.time*videoDuration(),type:'marker',pri:2,idx:i}));
  return pts;
}

function snapPosition(valSec,skip){
  let snapSec=snapThresholdPx*totalDuration()/totalWidth();
  let best=null,minD=Infinity;
  getSnapPoints().forEach(p=>{
    if(skip&&p.type==='clip'&&p.idx===skip.idx&&p.edge===skip.edge)return;
    let d=Math.abs(p.val-valSec);
    if(d<snapSec&&d<minD){best=p;minD=d;}
  });
  return best?best.val:valSec;
}

// ---- Ruler (50 proportional divisions, labels in seconds) ----
function drawRuler(){
  let w=twrap.clientWidth,dpr=devicePixelRatio;
  rcanvas.width=w*dpr;rcanvas.height=20*dpr;
  rcanvas.style.width=w+'px';rcanvas.style.height='20px';
  let tF=totalFrames(),fps=30;
  rctx.clearRect(0,0,rcanvas.width,rcanvas.height);
  rctx.fillStyle='#0d0d15';rctx.fillRect(0,0,rcanvas.width,rcanvas.height);

  // Track header strip (matches timeline header, so ticks align with track content)
  let hdr=32*dpr;
  rctx.fillStyle='#111118';rctx.fillRect(0,0,hdr,20*dpr);
  rctx.strokeStyle='#2a2a3a';rctx.lineWidth=1;
  rctx.beginPath();rctx.moveTo(hdr,0);rctx.lineTo(hdr,20*dpr);rctx.stroke();

  // ~50 major divisions, rounded to nice frame counts
  let rawInt=tF/50;
  let nice=[1,2,3,5,10,15,30,60,150,300,600,900,1800,3600,7200,18000,36000];
  let majorInt=1;
  for(let ni of nice){majorInt=ni;if(ni>=rawInt)break;}

  for(let f=0;f<=tF;f+=majorInt){
    let x=frameToX(f)*dpr;
    if(x<hdr-4*dpr||x>w*dpr+4*dpr)continue;
    let isSec=(f%fps===0);
    rctx.strokeStyle=isSec?'#777':'#444';rctx.lineWidth=1;
    rctx.beginPath();rctx.moveTo(x,isSec?6*dpr:12*dpr);rctx.lineTo(x,20*dpr);rctx.stroke();
    if(isSec||f===0||f===tF){
      let sec=f/fps;
      rctx.fillStyle='#aaa';rctx.font=(9*dpr)+'px monospace';
      let label;
      if(sec>=3600)label=Math.floor(sec/3600)+'h'+Math.floor(sec%3600/60).toString().padStart(2,'0')+'m';
      else if(sec>=60)label=Math.floor(sec/60)+'m'+Math.floor(sec%60)+'s';
      else label=sec.toFixed(0)+'s';
      rctx.fillText(label,x+2,7*dpr);
    }
  }
  // In marker (blue)
  if(inPoint!==null){
    let ix=secToX(inPoint*videoDuration())*dpr;
    rctx.fillStyle='#0af';rctx.beginPath();rctx.moveTo(ix,0);rctx.lineTo(ix-5*dpr,6*dpr);rctx.lineTo(ix+5*dpr,6*dpr);rctx.closePath();rctx.fill();
    rctx.fillStyle='#0af';rctx.font=(7*dpr)+'monospace';rctx.fillText('IN',ix+6*dpr,6*dpr);
  }
  // Out marker (orange)
  if(outPoint!==null){
    let ox=secToX(outPoint*videoDuration())*dpr;
    rctx.fillStyle='#f80';rctx.beginPath();rctx.moveTo(ox,0);rctx.lineTo(ox-5*dpr,6*dpr);rctx.lineTo(ox+5*dpr,6*dpr);rctx.closePath();rctx.fill();
    rctx.fillStyle='#f80';rctx.font=(7*dpr)+'monospace';rctx.fillText('OUT',ox+6*dpr,6*dpr);
  }
  // Marker ticks
  markers.forEach((m,i)=>{
    let mx=secToX(m.time*videoDuration())*dpr;
    rctx.fillStyle=m.color||'#ff0';rctx.beginPath();rctx.moveTo(mx,12*dpr);rctx.lineTo(mx-3*dpr,16*dpr);rctx.lineTo(mx+3*dpr,16*dpr);rctx.closePath();rctx.fill();
  });
}

// ---- Thumbnails (pre-generated sprite strip at ~1fps, cached in browser) ----
let _thumbH=0,_thumbN=0,_thumbBusy=false,_pregenStrip=null,_cacheKey=null;
function thumbAR(){return videoEl.videoWidth&&videoEl.videoHeight?videoEl.videoWidth/videoEl.videoHeight:16/9}

function _buildThumbsFromStrip(n,cw,ch){
  if(!_pregenStrip||!_pregenStrip.complete||!_pregenStrip.naturalWidth)return;
  // For cached strips, use the strip's stored totalFrames. Otherwise default to duration in seconds.
  var total=_pregenStrip._totalFrames || Math.ceil(videoEl.duration);
  if(total<1)return;
  var tileW=_pregenStrip.naturalWidth/total;
  thumbCache=Array(n).fill(null);
  for(var i=0;i<n;i++){
    var srcT=(i+0.5)/n*videoEl.duration;
    var srcIdx=Math.min(total-1,Math.floor(srcT));
    var c=document.createElement('canvas');
    c.width=cw;c.height=ch;
    c.getContext('2d').drawImage(_pregenStrip,Math.round(srcIdx*tileW),0,Math.round(tileW),ch,0,0,cw,ch);
    thumbCache[i]=c;
  }
  _bgDirty=true;drawTimeline();
}

function buildThumbnails(){
  if(!videoEl.duration)return;
  if(!_studioSid)return; // wait for backend upload
  let trkH=twrap.clientHeight/2-10;if(trkH<20)trkH=20;
  // Only rebuild on track height change; zoom never regenerates pre-rendered thumbs
  let ar=thumbAR();
  let thumbW=Math.round(trkH*ar);
  let clip0=clips[0],clipWPx=secToX(clip0.endSec)-secToX(clip0.startSec);
  let n=Math.max(1,Math.ceil(clipWPx/thumbW));
  if(thumbCache&&Math.abs(trkH-_thumbH)<8&&_thumbN===n)return;
  if(_thumbBusy)return;
  _thumbH=trkH;_thumbN=n;
  let srcH=Math.round(Math.max(90,trkH*devicePixelRatio));
  let cw=Math.round(srcH*ar),ch=srcH;

  if(_studioSid){
    // Use cached frames strip if available (fast, no ffmpeg)
    if(_cacheKey){
      if(!_pregenStrip||_pregenStrip._w!==cw||_pregenStrip._h!==ch){
        _thumbBusy=true;
        thumbCache=null;_bgDirty=true;drawTimeline();
        _pregenStrip=new Image();
        _pregenStrip._w=cw;_pregenStrip._h=ch;
        var totalNeeded=Math.ceil(videoEl.duration*2);  // 2fps cached frames
        _pregenStrip._totalFrames=totalNeeded;
        _pregenStrip.onload=function(){_thumbBusy=false;_buildThumbsFromStrip(n,cw,ch);};
        _pregenStrip.onerror=function(){console.warn('cached strip failed to load, disabling cache');_pregenStrip=null;_thumbBusy=false;_cacheKey=null;};
        _pregenStrip.src='/studio/frames-strip?cache_key='+_cacheKey+'&start=0&count='+totalNeeded+'&w='+cw+'&h='+ch;
        return;
      }
      _buildThumbsFromStrip(n,cw,ch);
    }else{
      // Pre-generated strip at ~1fps — loaded once, cached by browser, never re-fetched
      if(!_pregenStrip||_pregenStrip._w!==cw||_pregenStrip._h!==ch){
        _thumbBusy=true;
        thumbCache=null;_bgDirty=true;drawTimeline();
        _pregenStrip=new Image();
        _pregenStrip._w=cw;_pregenStrip._h=ch;
        _pregenStrip.onload=function(){_thumbBusy=false;_buildThumbsFromStrip(n,cw,ch);};
        _pregenStrip.onerror=function(){console.warn('pregenStrip failed to load');_pregenStrip=null;_thumbBusy=false;};
        _pregenStrip.src='/studio/thumbs-pregen?sid='+_studioSid+'&w='+cw+'&h='+ch;
        return;
      }
      _buildThumbsFromStrip(n,cw,ch);
    }
  }else{
    // No backend: hidden video element fallback
    _thumbBusy=true;
    thumbCache=Array(n).fill(null);_bgDirty=true;drawTimeline();
    var _tv=document.createElement('video');
    _tv.muted=true;_tv.preload='auto';
    _tv.src=videoEl.querySelector('source').src;_tv.load();
    var idx=0;
    function _nextThumb(){
      if(idx>=n){_thumbBusy=false;_bgDirty=true;drawTimeline();return;}
      _tv.currentTime=(idx+0.5)/n*videoEl.duration;
    }
    _tv.onseeked=function(){
      if(!thumbCache){_thumbBusy=false;return;} // aborted
      var c=document.createElement('canvas');c.width=cw;c.height=ch;
      c.getContext('2d').drawImage(_tv,0,0,cw,ch);
      thumbCache[idx]=c;idx++;_nextThumb();
    };
    _tv.onerror=function(){_thumbBusy=false;_bgDirty=true;drawTimeline();};
    _tv.onloadedmetadata=function(){_nextThumb();};
  }
}

// ---- Main timeline draw (cached background + dynamic overlay) ----
function drawTimeline(){
  let w=twrap.clientWidth,h=twrap.clientHeight,dpr=devicePixelRatio;
  if(!w||!h)return;
  let th=h/2,trkH=th-10;

  // ── Static background cache (grid, headers, clips, thumbnails, track 2) ──
  let needRebuild=_bgDirty||!_bgCache||_bgCache.width!==w*dpr||_bgCache.height!==h*dpr;
  if(needRebuild){
    if(!_bgCache)_bgCache=document.createElement('canvas');
    _bgCache.width=w*dpr;_bgCache.height=h*dpr;
    let bc=_bgCache.getContext('2d');
    let tW=totalWidth(),tF=totalFrames();
    let hdrW=32*dpr;

    // Grid lines
    let majorInt=Math.max(30,Math.ceil(tF/40)*30);
    for(let f=0;f<tF;f+=majorInt){
      let x=frameToX(f)*dpr;
      bc.strokeStyle='rgba(255,255,255,.04)';bc.lineWidth=1;
      bc.beginPath();bc.moveTo(x,0);bc.lineTo(x,h*dpr);bc.stroke();
    }

    // Track headers
    bc.fillStyle='#111118';bc.fillRect(0,0,hdrW,h*dpr);
    bc.strokeStyle='#2a2a3a';bc.lineWidth=1;
    bc.beginPath();bc.moveTo(hdrW,0);bc.lineTo(hdrW,h*dpr);bc.stroke();
    bc.fillStyle='#aaa';bc.font=(11*dpr)+'sans-serif';
    bc.fillText('V1',6*dpr,th/2*dpr+6*dpr);
    bc.fillText('FX',6*dpr,th*dpr+th/2*dpr+6*dpr);

    // Slider track (gray bar, static)
    let sliderBarY=Math.round(h*dpr*0.45),sliderBarH=Math.round(h*dpr*0.1);
    bc.fillStyle='rgba(255,255,255,.06)';
    bc.fillRect(hdrW,sliderBarY,_bgCache.width-hdrW,sliderBarH);

    // Clips on Track 1
    let ty0=6*dpr,th_px=trkH*dpr;
    clips.forEach((c,i)=>{
      let cx0=secToX(c.startSec),cx1=secToX(c.endSec);
      if(cx1-cx0<2){cx1=cx0+2;}
      let tx0=cx0*dpr,tw_px=(cx1-cx0)*dpr;
      let isSel=i===selectedClipIdx;
      let grad=bc.createLinearGradient(tx0,ty0,tx0+tw_px,ty0);
      grad.addColorStop(0,isSel?'#3a5a8a':'#1e283c');
      grad.addColorStop(1,isSel?'#2a4a7a':'#1a2230');
      bc.fillStyle=grad;bc.fillRect(tx0,ty0,tw_px,th_px);
      bc.strokeStyle=isSel?'#7eb8f7':'#4a7ab7';bc.lineWidth=(isSel?2:1.5)*dpr;
      bc.strokeRect(tx0,ty0,tw_px,th_px);
      // Thumbnails
      if(thumbCache&&thumbCache.filter(Boolean).length>0){
        let ar=thumbAR();
        let thW=Math.round(trkH*ar)*dpr;
        let n=thumbCache.length;
        bc.save();bc.beginPath();bc.rect(tx0,ty0,tw_px,th_px);bc.clip();
        for(let j=0;j<n;j++){
          if(thumbCache[j]){
            let dx=tx0+j*thW;
            if(dx>=tx0+tw_px)break;
            let dw=Math.min(thW,tx0+tw_px-dx);
            bc.drawImage(thumbCache[j],0,0,Math.round(thumbCache[j].width*dw/thW),thumbCache[j].height,dx,ty0,dw,th_px);
          }
        }
        bc.restore();
      }else{
        bc.fillStyle='#555';bc.font=(10*dpr)+'monospace';
        bc.fillText(c.label||'Clip',tx0+4,ty0+th_px/2+4*dpr);
      }
      // Edge handles
      bc.fillStyle=isSel?'#fff':'#7eb8f7';
      bc.fillRect(tx0-5*dpr,ty0,9*dpr,th_px);
      bc.fillRect(tx0+tw_px-4*dpr,ty0,9*dpr,th_px);
      // Duration label
      let dur=c.endSec-c.startSec;
      bc.fillStyle='rgba(255,255,255,.7)';bc.font=(9*dpr)+'monospace';
      bc.fillText(dur.toFixed(1)+'s',tx0+tw_px/2-15*dpr,ty0+th_px-4*dpr);
      // Label
      if(c.label){
        bc.fillStyle='rgba(200,200,220,.8)';bc.font=(9*dpr)+'monospace';
        bc.fillText(c.label,tx0+4,ty0-3*dpr);
      }
    });

    // Track 2: Face detection
    let ty2=th*dpr+6*dpr,th2=trkH*dpr;
    if(window._faceData){
      bc.fillStyle='rgba(46,139,87,.5)';
      (window._faceData.regions||[]).forEach(([s,e])=>{
        let c0=clips[0],absS=c0.startSec+(c0.endSec-c0.startSec)*s;
        let ww=(c0.endSec-c0.startSec)*(e-s)*zoomLevel/100;
        bc.fillRect(secToX(absS)*dpr,ty2,ww*dpr,th2);
      });
      bc.fillStyle='rgba(178,34,34,.4)';
      (window._faceData.gaps||[]).forEach(([s,e])=>{
        let c0=clips[0],absS=c0.startSec+(c0.endSec-c0.startSec)*s;
        let ww=(c0.endSec-c0.startSec)*(e-s)*zoomLevel/100;
        bc.fillRect(secToX(absS)*dpr,ty2,ww*dpr,th2);
      });
    }else if(clips.length>0){
      let c0=clips[0],cx0=secToX(c0.startSec),cx1=secToX(c0.endSec);
      bc.fillStyle='rgba(60,60,60,.4)';
      bc.fillRect(cx0*dpr,ty2,(cx1-cx0)*dpr,th2);
    }

    _bgDirty=false;
  }

  // ── Blit cached background ──
  tcanvas.width=w*dpr;tcanvas.height=h*dpr;
  tcanvas.style.width=w+'px';tcanvas.style.height=h+'px';
  tctx.clearRect(0,0,tcanvas.width,tcanvas.height);
  tctx.drawImage(_bgCache,0,0);

  // ── Dynamic overlay (redrawn every call) ──
  let hdrW=32*dpr;
  let sliderBarY=Math.round(h*dpr*0.45),sliderBarH=Math.round(h*dpr*0.1);

  // In/Out range highlight on slider + full-height dashed lines
  if(inPoint!==null&&outPoint!==null){
    let lo=Math.min(inPoint,outPoint),hi=Math.max(inPoint,outPoint);
    let ix=secToX(lo*videoDuration()),ox=secToX(hi*videoDuration());
    let lerp=(x)=>{return hdrW+(tcanvas.width-hdrW)*x};
    let leftX=Math.max(ix*dpr,lerp(0)),rightX=Math.min(ox*dpr,lerp(1));
    tctx.fillStyle='rgba(0,170,255,.35)';
    tctx.fillRect(leftX,sliderBarY,rightX-leftX,sliderBarH);
    tctx.strokeStyle='rgba(0,170,255,.5)';tctx.lineWidth=2*dpr;
    tctx.setLineDash([4*dpr,3*dpr]);
    tctx.beginPath();tctx.moveTo(ix*dpr,0);tctx.lineTo(ix*dpr,h*dpr);tctx.stroke();
    tctx.beginPath();tctx.moveTo(ox*dpr,0);tctx.lineTo(ox*dpr,h*dpr);tctx.stroke();
    tctx.setLineDash([]);
  }
  // In handle (blue grip)
  if(inPoint!==null){
    let ix=secToX(inPoint*videoDuration())*dpr;
    tctx.fillStyle='#0af';
    tctx.fillRect(ix-4*dpr,0,8*dpr,16*dpr);
    tctx.fillStyle='rgba(0,170,255,.12)';
    tctx.fillRect(ix-2*dpr,16*dpr,4*dpr,h*dpr-16*dpr);
  }
  // Out handle (orange grip)
  if(outPoint!==null){
    let ox=secToX(outPoint*videoDuration())*dpr;
    tctx.fillStyle='#f80';
    tctx.fillRect(ox-4*dpr,0,8*dpr,16*dpr);
    tctx.fillStyle='rgba(255,136,0,.12)';
    tctx.fillRect(ox-2*dpr,16*dpr,4*dpr,h*dpr-16*dpr);
  }

  // Playhead
  if(videoEl.duration){
    let px=secToX(videoEl.currentTime)*dpr;
    tctx.strokeStyle='#f44';tctx.lineWidth=2*dpr;
    tctx.beginPath();tctx.moveTo(px,0);tctx.lineTo(px,h*dpr);tctx.stroke();
    tctx.fillStyle='#f44';
    tctx.beginPath();tctx.moveTo(px,0);tctx.lineTo(px-5*dpr,8*dpr);tctx.lineTo(px+5*dpr,8*dpr);tctx.closePath();tctx.fill();
  }

  // Cyan hover indicator (follows mouse)
  if(!dragging&&mouseTX>32){
    tctx.strokeStyle='rgba(0,200,220,.25)';tctx.lineWidth=1*dpr;
    tctx.beginPath();tctx.moveTo(mouseTX*dpr,0);tctx.lineTo(mouseTX*dpr,h*dpr);tctx.stroke();
  }

  drawRuler();
  // Update cursor overlay div
  if(videoEl.duration){
    tcursor.style.left=(secToX(videoEl.currentTime)/w*100)+'%';
  }
}

function drawTimelineLoop(){
  if(videoEl.duration){
    buildThumbnails();
    // Lightweight: just move CSS overlay playhead, no canvas redraw
    let w=twrap.clientWidth;
    if(w>0){tcursor.style.left=(secToX(videoEl.currentTime)/w*100)+'%';}
  }
  requestAnimationFrame(drawTimelineLoop);
}

// ---- Zoom at cursor (PR-style: keep cursor-anchored frame fixed) ----
function zoomAtCursor(cursorX,factor){
  let oldF=totalFrames(),oldW=totalWidth();
  let cursorFrame=(cursorX+scrollX-32)*oldF/oldW;
  if(cursorFrame<0)cursorFrame=0;
  zoomLevel=Math.max(1,Math.min(500,zoomLevel*factor));
  document.getElementById('zoom-slider').value=zoomLevel;
  let newF=totalFrames(),newW=totalWidth();
  scrollX=32+cursorFrame*newW/newF-cursorX;
  clampScroll();
  _bgDirty=true;drawTimeline();
}

function handleTimelineWheel(e){
  e.preventDefault();
  let rect=twrap.getBoundingClientRect();
  if(e.ctrlKey||e.metaKey){
    zoomAtCursor(e.clientX-rect.left,e.deltaY>0?.85:1.18);
  }else{
    scrollX+=e.deltaY*3;
    clampScroll();
    _bgDirty=true;drawTimeline();
  }
}

// ---- Hit testing ----
function _hitTest(e){
  let rect=twrap.getBoundingClientRect(),x=e.clientX-rect.left;
  // In/out drag handles
  if(videoEl.duration){
    if(inPoint!==null){
      let ix=secToX(inPoint*videoDuration());
      if(Math.abs(x-ix)<14)return{type:'in-drag'};
    }
    if(outPoint!==null){
      let ox=secToX(outPoint*videoDuration());
      if(Math.abs(x-ox)<14)return{type:'out-drag'};
    }
    // Red playhead (draggable)
    let phX=secToX(videoEl.currentTime);
    if(Math.abs(x-phX)<8)return{type:'playhead'};
  }
  return{type:'empty'};
}

// ---- Mouse handlers (single-click to seek, drag only in/out handles) ----
function _onDragMove(e){
  if(!dragging)return;
  if(Math.abs(e.clientX-dragStartX)>3)_wasDrag=true;
  let dx=e.clientX-dragStartX;
  let ds=dx*totalDuration()/totalWidth();
  let dur=videoDuration();
  if(dragging.type==='in-drag'){
    let val=Math.max(0,Math.min(1,(dragStartVal*dur+ds)/dur));
    if(outPoint!==null&&val>=outPoint)val=outPoint-1/fps/dur;
    inPoint=val;
    _setCurrentTime(inPoint*dur);
    drawTimeline();
  }else if(dragging.type==='out-drag'){
    let val=Math.max(0,Math.min(1,(dragStartVal*dur+ds)/dur));
    if(inPoint!==null&&val<=inPoint)val=inPoint+1/fps/dur;
    outPoint=val;
    _setCurrentTime(outPoint*dur);
    drawTimeline();
  }else if(dragging.type==='playhead'){
    let t=Math.max(0,Math.min(dur,dragStartVal+ds));
    _setCurrentTime(t);
    drawTimeline();
  }
}
function _onDragUp(e){
  document.removeEventListener('mousemove',_onDragMove);
  document.removeEventListener('mouseup',_onDragUp);
  dragging=null;
}
function handleTimelineMouseDown(e){
  _wasDrag=false;
  let hit=_hitTest(e);
  if(hit.type==='in-drag'||hit.type==='out-drag'||hit.type==='playhead'){
    dragging=hit;dragStartX=e.clientX;
    if(hit.type==='in-drag')dragStartVal=inPoint;
    else if(hit.type==='out-drag')dragStartVal=outPoint;
    else dragStartVal=videoEl.currentTime;
    document.addEventListener('mousemove',_onDragMove);
    document.addEventListener('mouseup',_onDragUp);
    e.preventDefault();return;
  }
  dragging=null;
  e.preventDefault();
}


// Click on track → move playhead (single click, not drag)
twrap.addEventListener('click',function(e){
  if(_wasDrag||!videoEl.duration)return;
  let rect=twrap.getBoundingClientRect(),x=e.clientX-rect.left;
  if(x>32){
    _setCurrentTime(Math.max(0,xToSec(x)));
    drawTimeline();
  }
});

// Right-click on track → set in/out point
twrap.addEventListener('contextmenu',function(e){
  e.preventDefault();
  if(!videoEl.duration)return;
  let rect=twrap.getBoundingClientRect(),x=e.clientX-rect.left;
  if(x<32)return;
  let t=Math.max(0,xToSec(x))/videoDuration();
  if(e.shiftKey){
    outPoint=Math.min(1,t);
    if(inPoint!==null&&outPoint<=inPoint)outPoint=Math.min(1,inPoint+1/fps/videoDuration());
  }else{
    inPoint=Math.max(0,t);
    if(outPoint!==null&&inPoint>=outPoint)inPoint=Math.max(0,outPoint-1/fps/videoDuration());
    _setCurrentTime(t*videoDuration());
  }
  drawTimeline();
});

// ---- Keyboard shortcuts (PR-style) ----
document.addEventListener('keydown',function(e){
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
  if(!videoEl.duration)return;
  let fps=30;
  switch(e.key){
    case'ArrowLeft':
      _setCurrentTime(videoEl.currentTime-(e.shiftKey?10:1)/fps);
      e.preventDefault();drawTimeline();break;
    case'ArrowRight':
      _setCurrentTime(videoEl.currentTime+(e.shiftKey?10:1)/fps);
      e.preventDefault();drawTimeline();break;
    case'i':case'I':
      if(e.shiftKey){if(inPoint!==null)seekVideo(Math.max(0,Math.min(1,inPoint)));e.preventDefault();break;}
      if(e.altKey||(e.ctrlKey&&e.shiftKey)){inPoint=null;e.preventDefault();drawTimeline();break;}
      inPoint=videoEl.currentTime/videoEl.duration;
      e.preventDefault();drawTimeline();break;
    case'o':case'O':
      if(e.shiftKey){if(outPoint!==null)seekVideo(Math.max(0,Math.min(1,outPoint)));e.preventDefault();break;}
      if(e.altKey||(e.ctrlKey&&e.shiftKey)){outPoint=null;e.preventDefault();drawTimeline();break;}
      outPoint=videoEl.currentTime/videoEl.duration;
      e.preventDefault();drawTimeline();break;
    case's':case'S':
      if(e.ctrlKey||e.metaKey)break;
      if(selectedClipIdx>=0){
        let c=clips[selectedClipIdx];
        let tSec=videoEl.currentTime;
        if(tSec>c.startSec+1/30&&tSec<c.endSec-1/30){
          let rightPart={startSec:tSec,endSec:c.endSec,label:(c.label||'Clip')+'_2',color:c.color,track:c.track};
          c.endSec=tSec;
          clips.splice(selectedClipIdx+1,0,rightPart);
          selectedClipIdx=selectedClipIdx+1;
          e.preventDefault();_bgDirty=true;drawTimeline();
        }
      }
      break;
    case'm':case'M':
      {let mt=videoEl.currentTime/videoEl.duration;
      let ex=markers.findIndex(m=>Math.abs(m.time-mt)<.0008);
      if(ex>=0)markers.splice(ex,1);
      else markers.push({time:mt,label:'M'+(markers.length+1),color:'#ff0'});
      e.preventDefault();drawTimeline();}
      break;
    case'Escape':
      selectedClipIdx=-1;
      e.preventDefault();_bgDirty=true;drawTimeline();break;
    case'j':case'J':
      videoEl.playbackRate=Math.max(-4,(videoEl.playbackRate||1)-1);videoEl.play();e.preventDefault();break;
    case'k':case'K':
      videoEl.pause();videoEl.playbackRate=1;e.preventDefault();break;
    case'l':case'L':
      videoEl.playbackRate=Math.min(4,(videoEl.playbackRate||1)+1);videoEl.play();e.preventDefault();break;
    case' ':
      e.preventDefault();
      if(videoEl.paused)videoEl.play();else videoEl.pause();
      break;
  }
});

// Loop between in/out during playback (merged into onVideoTimeUpdate)
function onVideoTimeUpdate(){
  const t=videoEl.currentTime;
  const ts=_fmtTime(t);
  document.getElementById('timeline-pos').textContent=ts;
  document.getElementById('preview-time').textContent=ts;
  // Loop between in/out
  if(inPoint!==null&&outPoint!==null&&!videoEl.paused){
    let lo=Math.min(inPoint,outPoint),hi=Math.max(inPoint,outPoint);
    let loT=lo*videoEl.duration,hiT=hi*videoEl.duration;
    if(t>=hiT)videoEl.currentTime=loT;
  }
}

// ---- Zoom controls ----
document.getElementById('zoom-slider').addEventListener('input',function(){
  zoomLevel=parseInt(this.value);_bgDirty=true;drawTimeline();
});
document.getElementById('btn-zoom-out').addEventListener('click',function(){
  zoomAtCursor(twrap.clientWidth/2,.8);
});
document.getElementById('btn-zoom-in').addEventListener('click',function(){
  zoomAtCursor(twrap.clientWidth/2,1.25);
});

// Ruler click: jump to nearest frame
rcanvas.addEventListener('click',function(e){
  if(!videoEl.duration)return;
  let rect=rcanvas.getBoundingClientRect(),x=e.clientX-rect.left;
  let sec=Math.max(0,xToSec(x));
  _setCurrentTime(sec);
  if(!e.shiftKey){selectedClipIdx=-1;_bgDirty=true;}
  drawTimeline();
});

window.addEventListener('resize',function(){_bgDirty=true;drawTimeline();});
drawTimelineLoop();

// ========== Timeline resize ==========
const tresizeHandle=document.getElementById('timeline-resize-handle');
const tlineWrap=document.getElementById('timeline-wrap');
let tResizing=false,tResizeStart=0,tResizeStartH=0;
tresizeHandle.addEventListener('mousedown',e=>{
  tResizing=true;tResizeStart=e.clientY;
  tResizeStartH=parseInt(getComputedStyle(tlineWrap).height);
  e.preventDefault();
});
document.addEventListener('mousemove',e=>{
  if(!tResizing)return;
  let nh=Math.max(100,Math.min(300,tResizeStartH+(tResizeStart-e.clientY)));
  tlineWrap.style.height=nh+'px';
  _bgDirty=true;drawTimeline();
});
document.addEventListener('mouseup',()=>{tResizing=false});

// ========== Resizable panels ==========
let resizeActive=null,resizeStartX=0,resizeStartW=0;
document.querySelectorAll('.resize-handle').forEach(h=>{
  h.addEventListener('mousedown',e=>{
    resizeActive=h;resizeStartX=e.clientX;h.classList.add('active');
    const target=h.dataset.target==='left'?'left-panel':'right-panel';
    resizeStartW=parseInt(getComputedStyle(document.getElementById(target)).width);
    e.preventDefault();
  });
});
document.addEventListener('mousemove',e=>{
  if(!resizeActive)return;
  const targetId=resizeActive.dataset.target==='left'?'left-panel':'right-panel';
  const target=document.getElementById(targetId);
  const dx=e.clientX-resizeStartX;
  const newW=Math.max(targetId==='left-panel'?160:200,resizeStartW+(resizeActive.dataset.target==='left'?dx:-dx));
  target.style.width=newW+'px';
  target.style.flexShrink='0';
  _bgDirty=true;drawTimeline();
});
document.addEventListener('mouseup',()=>{
  if(resizeActive){resizeActive.classList.remove('active');resizeActive=null;}
});

// Ctrl+scroll zoom (prevent browser zoom conflict)
document.addEventListener('wheel',function(e){
  if(e.ctrlKey||e.metaKey){e.preventDefault()}
},{passive:false});

// ========== Button handlers ==========
function onRenderCache(ev){
  var btn = ev.target;
  if(!_studioSid){ alert('请先加载视频'); return; }
  btn.disabled = true; btn.textContent = '⏳ 提取中...';
  fetch('/studio/extract-cache', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({sid:_studioSid})
  }).then(function(r){return r.json();}).then(function(d){
    btn.disabled = false;
    if(d.status === 'done' || d.status === 'cached'){
      btn.textContent = '✅ 已缓存';
      // Force full thumbnail rebuild from backend strip
      thumbCache = null; _pregenStrip = null; _thumbH = 0; _thumbN = 0; _bgDirty = true;
      // Extract cache key from response path: studio_cache/<cache_key>/
      if(d.cache_dir){_cacheKey = d.cache_dir.split(/[/\\\\]/).pop();}
      drawTimeline();
      // Refresh cache list UI
      if(typeof loadCacheList === 'function') setTimeout(loadCacheList, 200);
      // Also re-trigger buildThumbnails immediately
      setTimeout(function(){ buildThumbnails(); drawTimeline(); }, 100);
    }else{
      btn.textContent = '&#128190; 渲染帧缓存';
      alert('提取失败: ' + (d.error || '未知错误'));
    }
  }).catch(function(e){
    btn.disabled = false;
    btn.textContent = '&#128190; 渲染帧缓存';
    alert('请求失败: ' + e.message);
  });
}

function onExportFrame(ev){
  var btn = ev.target;
  if(!_studioSid || !videoEl.duration){ alert('请先加载视频'); return; }
  btn.disabled = true; btn.textContent = '⏳...';
  fetch('/studio/export-frame', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({sid:_studioSid, time_sec:videoEl.currentTime})
  }).then(function(r){
    if(!r.ok) throw new Error('export failed');
    return r.blob();
  }).then(function(blob){
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    var name = _currentVideoFile ? _currentVideoFile.name.replace(/\.[^.]+$/,'') : 'frame';
    name += '_' + _fmtTime(videoEl.currentTime).replace(/:/g,'-') + '.jpg';
    a.download = name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function(){URL.revokeObjectURL(a.href);}, 1000);
    btn.disabled = false; btn.textContent = '&#128196; 导出当前帧';
  }).catch(function(e){
    btn.disabled = false; btn.textContent = '&#128196; 导出当前帧';
    alert('导出失败: ' + e.message);
  });
}

var _exportTaskId = null;
var _exportPollTimer = null;
var _exportStartSec = 0;
var _exportEndSec = 0;

function onBatchExport(){
  if(!_studioSid || !videoEl.duration){ alert('请先加载视频'); return; }
  if(inPoint === null || outPoint === null){ alert('请先在时间轴上设置起止范围（右键设置 in/out）'); return; }
  var dur = videoEl.duration;
  _exportStartSec = Math.min(inPoint, outPoint) * dur;
  _exportEndSec = Math.max(inPoint, outPoint) * dur;
  var totalFrames = Math.ceil((_exportEndSec - _exportStartSec) * 30);

  // Fill in the Step 0 dialog
  document.getElementById('exp-start-time').textContent = _exportStartSec.toFixed(1) + 's (' + _fmtTime(_exportStartSec) + ')';
  document.getElementById('exp-end-time').textContent = _exportEndSec.toFixed(1) + 's (' + _fmtTime(_exportEndSec) + ')';
  document.getElementById('exp-total-frames').textContent = totalFrames;
  document.getElementById('exp-jpeg-quality').value = 95;
  document.getElementById('exp-need-filter').checked = false;
  // Pre-fill output directory
  var defaultDir = (_currentVideoFile ? _currentVideoFile.name.replace(/\.[^.]+$/,'') : 'export') + '_frames';
  document.getElementById('exp-output-dir').value = 'C:\\Users\\nobody\\Desktop\\' + defaultDir;

  // Show confirmation dialog
  document.getElementById('export-dialog-overlay').style.display = 'flex';
}

function onExportConfirm(){
  var format = document.getElementById('exp-format').value;
  var quality = parseInt(document.getElementById('exp-jpeg-quality').value) || 95;
  var needFilter = document.getElementById('exp-need-filter').checked;
  var outputDir = document.getElementById('exp-output-dir').value.trim();
  if(!outputDir){ alert('请填写输出目录'); return; }

  // Hide dialog
  document.getElementById('export-dialog-overlay').style.display = 'none';

  var overlay = document.getElementById('progress-overlay');
  var fill = document.getElementById('progress-bar-fill');
  var phase = document.getElementById('progress-phase');
  var text = document.getElementById('progress-text');
  var timeEl = document.getElementById('progress-time');
  overlay.style.display = 'flex';
  fill.style.width = '0%';
  phase.textContent = '准备中...';
  text.textContent = '';
  timeEl.textContent = '';

  _exportTaskId = null;
  _exportPollTimer = null;

  function cancelExport(){
    if(_exportTaskId){
      fetch('/studio/cancel-batch-export-v2', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({task_id:_exportTaskId})});
    }
    overlay.style.display = 'none';
    document.getElementById('waiting-ui-overlay').style.display = 'none';
    if(_exportPollTimer) clearInterval(_exportPollTimer);
  }
  document.getElementById('progress-cancel').onclick = cancelExport;

  fetch('/studio/batch-export-v2', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({
    sid:_studioSid, start_sec:_exportStartSec, end_sec:_exportEndSec,
    output_format:format, jpeg_quality:quality, need_pause_for_filter:needFilter,
    output_dir:outputDir
  })}).then(function(r){return r.json();}).then(function(d){
    if(d.error){ alert('导出失败: '+d.error); overlay.style.display='none'; return; }
    _exportTaskId = d.task_id;
    _exportPollTimer = setInterval(function(){
      fetch('/studio/batch-export-status-v2?task_id='+_exportTaskId).then(function(r){return r.json();}).then(function(s){
        // Update three stage progress bars
        var stages = s.stages || [];
        var stageLabels = ['提取帧', '提取人脸', '换脸合成'];
        var currentStage = s.current_stage || 0;
        for(var i=0;i<3;i++){
          var row=document.getElementById('st'+i);
          var fill=document.getElementById('st'+i+'-fill');
          var pct=document.getElementById('st'+i+'-pct');
          var label=document.getElementById('st'+i+'-label');
          if(!row)continue;
          row.className='stage-row';
          label.className='stage-label';
          if(i < currentStage){ row.classList.add('stage-done'); label.classList.add('done'); }
          else if(i === currentStage){ row.classList.add('stage-active'); label.classList.add('active'); }
          else { row.classList.add('stage-pending'); }
          if(stages[i]){
            var p=stages[i].progress||0;
            var curr=stages[i].current||0;
            var tot=stages[i].total||0;
            fill.style.width=(p*100)+'%';
            pct.textContent=tot>0?curr+'/'+tot:(p>0?Math.round(p*100)+'%':'-');
          }
        }

        // Handle waiting_for_user — show Step 2.5 UI
        if(s.status === 'waiting_for_user'){
          overlay.style.display = 'none';
          document.getElementById('waiting-dir-path').textContent = s.aligned_dir || '--';
          document.getElementById('waiting-ui-overlay').style.display = 'flex';
          return;
        }

        if(s.status === 'done' || s.status === 'cancelled'){
          clearInterval(_exportPollTimer);
          document.getElementById('waiting-ui-overlay').style.display = 'none';
          if(s.status === 'done'){
            for(var i=0;i<3;i++){
              var f=document.getElementById('st'+i+'-fill');
              if(f)f.style.width='100%';
            }
            phase.textContent = '✅ 导出完成！';
            text.textContent = '共 ' + (s.current_frame || s.total_frames || '?') + ' 帧，用时 ' + Math.round(s.elapsed_sec || 0) + ' 秒';
            timeEl.textContent = '';
            setTimeout(function(){ overlay.style.display = 'none'; }, 3000);
          } else {
            overlay.style.display = 'none';
          }
          return;
        }
        if(s.status === 'error'){
          phase.textContent = '❌ 导出失败';
          text.textContent = s.error || '';
          timeEl.textContent = '';
          return;
        }
        fill.style.width = (s.progress * 100) + '%';
        text.textContent = (s.current_frame || '0') + ' / ' + (s.total_frames || '?') + ' (' + Math.round(s.progress*100) + '%)';
        if(s.elapsed_sec > 0 && s.progress > 0){
          var remaining = (s.elapsed_sec / s.progress) - s.elapsed_sec;
          timeEl.textContent = '已用 ' + Math.round(s.elapsed_sec) + 's，预计剩余 ' + Math.round(remaining) + 's';
        } else {
          timeEl.textContent = '';
        }
      }).catch(function(){});
    }, 500);
  }).catch(function(e){
    alert('请求失败: '+e.message);
    overlay.style.display = 'none';
  });
}

function onContinueMerge(){
  document.getElementById('waiting-ui-overlay').style.display = 'none';
  var overlay = document.getElementById('progress-overlay');
  overlay.style.display = 'flex';
  document.getElementById('progress-phase').textContent = '合成中...';
  document.getElementById('progress-bar-fill').style.width = '0%';
  document.getElementById('progress-text').textContent = '';
  document.getElementById('progress-time').textContent = '';

  fetch('/studio/continue-batch-export', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({task_id:_exportTaskId})})
    .then(function(r){return r.json();}).then(function(d){
      if(d.error){ alert('继续合成失败: '+d.error); overlay.style.display='none'; return; }
    }).catch(function(e){
      alert('请求失败: '+e.message);
      overlay.style.display='none';
    });
}

function onCancelExport(){
  document.getElementById('waiting-ui-overlay').style.display = 'none';
  document.getElementById('progress-overlay').style.display = 'none';
  if(_exportTaskId){
    fetch('/studio/cancel-batch-export-v2', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({task_id:_exportTaskId})});
  }
  if(_exportPollTimer) clearInterval(_exportPollTimer);
}

// ========== Cache management ==========
function loadCacheList(){
  fetch('/studio/cache-list').then(function(r){return r.json();}).then(function(list){
    var el = document.getElementById('cache-list');
    if(!el) return;
    if(!list || list.length === 0){
      el.innerHTML = '<div class="cache-empty">暂无缓存</div>';
      return;
    }
    var totalSize = 0;
    var html = '';
    list.forEach(function(item){
      totalSize += item.size_bytes;
      html += '<div class="cache-item">';
      html += '<div class="info"><div class="name">' + item.filename + '</div><div class="detail">' + item.total_frames + ' 帧 · ' + item.size_mb + ' MB</div></div>';
      html += '<span class="del" data-key="' + item.key + '" onclick="deleteCacheItem(this)">✕</span>';
      html += '</div>';
    });
    html += '<div class="cache-total">总计: ' + (totalSize / 1048576).toFixed(1) + ' MB</div>';
    el.innerHTML = html;
  }).catch(function(){});
}

function deleteCacheItem(el){
  var key = el.getAttribute('data-key');
  if(!confirm('删除缓存 "' + key + '"？')) return;
  fetch('/studio/cache-delete', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({key:key})})
    .then(function(r){return r.json();}).then(function(d){
      if(d.deleted) loadCacheList();
    }).catch(function(){});
}

// Load cache list initially and when right panel becomes visible
setTimeout(loadCacheList, 1000);

// ========== Settings sync ==========
function _gatherSettings(){
  var g = function(id){return document.getElementById(id);};
  var val = function(id){var e=g(id);return e?e.value:null;};
  var num = function(id){var v=val(id);return v?parseInt(v):0;};
  var chk = function(id){var e=g(id);return e?e.checked:false;};
  return {
    device: val('cfg-device') || '0',
    preview_scale: num('cfg-preview-scale') || 2,
    detector: val('cfg-detector') || 'S3FD',
    landmark: 'insightface106pt2d',
    max_faces: num('cfg-max-faces') || 1,
    face_type: val('cfg-face-type') || 'full_face',
    mode: val('cfg-mode') || 'overlay',
    masked_hist_match: chk('cfg-masked-hist-match'),
    mask_mode: num('cfg-mask-mode'),
    erode_mask_modifier: num('cfg-erode-mask'),
    blur_mask_modifier: num('cfg-blur-mask'),
    motion_blur_power: num('cfg-motion-blur'),
    color_transfer_mode: val('cfg-color-transfer') || 'rct',
    output_face_scale: num('cfg-face-scale'),
    super_resolution_power: num('cfg-super-res'),
    image_denoise_power: num('cfg-denoise'),
    bicubic_degrade_power: num('cfg-bicubic'),
    color_degrade_power: num('cfg-color-degrade'),
    sharpen_mode: num('cfg-sharpen-mode'),
    blursharpen_amount: num('cfg-sharpen-amount'),
  };
}

function _pushSettings(){
  fetch('/studio/settings', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(_gatherSettings())
  }).catch(function(e){console.warn('settings sync failed',e);});
}

// Real-time monitor: push settings + re-analyze on any param change
var _paramDebounce = null;
document.querySelectorAll('#right-panel select, #right-panel input, #right-panel textarea').forEach(function(el){
  var handler = function(){
    _pushSettings();
    _lastAnalyzeTime = -1;
    if(videoEl.paused && videoEl.duration){
      if(_paramDebounce) clearTimeout(_paramDebounce);
      _paramDebounce = setTimeout(function(){ recompositePreview(); }, 300);
    }
  };
  el.addEventListener('input', handler);
  el.addEventListener('change', handler);
});

// Load settings on page load
function _applySettings(s){
  var setVal = function(id, val){var e=document.getElementById(id);if(e && val!==undefined && val!==null) e.value=val;};
  var setChk = function(id, val){var e=document.getElementById(id);if(e) e.checked=!!val;};
  setVal('cfg-device', s.device);
  setVal('cfg-preview-scale', String(s.preview_scale));
  setVal('cfg-detector', s.detector);
  setVal('cfg-max-faces', String(s.max_faces));
  setVal('cfg-face-type', s.face_type);
  setVal('cfg-mode', s.mode);
  setChk('cfg-masked-hist-match', s.masked_hist_match);
  setVal('cfg-mask-mode', String(s.mask_mode));
  setVal('cfg-erode-mask', String(s.erode_mask_modifier));
  setVal('cfg-blur-mask', String(s.blur_mask_modifier));
  setVal('cfg-motion-blur', String(s.motion_blur_power));
  setVal('cfg-color-transfer', s.color_transfer_mode);
  setVal('cfg-face-scale', String(s.output_face_scale));
  setVal('cfg-super-res', String(s.super_resolution_power));
  setVal('cfg-denoise', String(s.image_denoise_power));
  setVal('cfg-bicubic', String(s.bicubic_degrade_power));
  setVal('cfg-color-degrade', String(s.color_degrade_power));
  setVal('cfg-sharpen-mode', String(s.sharpen_mode));
  setVal('cfg-sharpen-amount', String(s.blursharpen_amount));
}
fetch('/studio/settings').then(function(r){return r.json();}).then(_applySettings).catch(function(){});
// ========== Three-panel preview ==========
var _lastAnalyzeTime = -1;

function _handleAnalyzeResponse(d){
  document.getElementById('detection-loading').style.display = 'none';
  document.getElementById('swap-loading').style.display = 'none';
  if(d.error){
    console.warn('analyze error:', d.error);
    document.getElementById('detection-loading').textContent = d.error;
    document.getElementById('detection-loading').style.display = 'flex';
    setTimeout(function(){
      document.getElementById('detection-loading').style.display = 'none';
      document.getElementById('detection-loading').textContent = '分析中...';
    }, 3000);
    return;
  }
  if(d.detection){
    var detCanvas = document.getElementById('preview-detection');
    var detImg = new Image();
    detImg.onload = function(){
      detCanvas.width = detImg.naturalWidth;
      detCanvas.height = detImg.naturalHeight;
      detCanvas.getContext('2d').drawImage(detImg, 0, 0);
    };
    detImg.src = 'data:image/jpeg;base64,' + d.detection;
  }
  if(d.swapped){
    var swapCanvas = document.getElementById('preview-swapped');
    var swapImg = new Image();
    swapImg.onload = function(){
      swapCanvas.width = swapImg.naturalWidth;
      swapCanvas.height = swapImg.naturalHeight;
      swapCanvas.getContext('2d').drawImage(swapImg, 0, 0);
    };
    swapImg.src = 'data:image/jpeg;base64,' + d.swapped;
  }
  if(!d.has_face){
    document.getElementById('detection-loading').textContent = '未检测到人脸';
    document.getElementById('detection-loading').style.display = 'flex';
    setTimeout(function(){
      document.getElementById('detection-loading').style.display = 'none';
      document.getElementById('detection-loading').textContent = '分析中...';
    }, 2000);
  }
  if(d.has_face && d.model_loaded === false){
    var sw = document.getElementById('swap-loading');
    sw.textContent = '请先加载DFM模型';
    sw.style.display = 'flex';
    setTimeout(function(){sw.style.display='none';sw.textContent='合成中...';}, 4000);
  }
}

function triggerAnalyzeFrame(){
  if(!videoEl.duration || !videoEl.paused) return;
  if(!_studioSid){
    document.getElementById('detection-loading').textContent = '等待视频上传...';
    document.getElementById('detection-loading').style.display = 'flex';
    return;
  }
  var t = videoEl.currentTime;
  if(Math.abs(t - _lastAnalyzeTime) < 0.1) return;
  _lastAnalyzeTime = t;

  document.getElementById('detection-loading').textContent = '分析中...';
  document.getElementById('detection-loading').style.display = 'flex';
  document.getElementById('swap-loading').style.display = 'flex';

  fetch('/studio/analyze-frame', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({sid:_studioSid, time_sec:t})
  }).then(function(r){return r.json();}).then(_handleAnalyzeResponse)
    .catch(function(e){
      document.getElementById('detection-loading').style.display = 'none';
      document.getElementById('swap-loading').style.display = 'none';
      console.warn('analyze-frame request failed:', e);
    });
}

function recompositePreview(){
  if(!_studioSid || !videoEl.duration || !videoEl.paused) return;
  _lastAnalyzeTime = videoEl.currentTime;
  document.getElementById('swap-loading').style.display = 'flex';
  var reqBody = JSON.stringify({sid:_studioSid, time_sec:videoEl.currentTime, settings:_gatherSettings()});
  fetch('/studio/recomposite', {method:'POST', headers:{'Content-Type':'application/json'}, body:reqBody})
    .then(function(r){return r.json();})
    .then(function(d){
      document.getElementById('swap-loading').style.display = 'none';
      if(d.swapped){
        var c=document.getElementById('preview-swapped');
        var img=new Image();
        img.onload=function(){c.width=img.naturalWidth;c.height=img.naturalHeight;c.getContext('2d').drawImage(img,0,0);};
        img.src='data:image/jpeg;base64,'+d.swapped;
      }
    })
    .catch(function(e){console.warn('recomposite failed:',e);});
}

videoEl.addEventListener('pause', function(){
  setTimeout(triggerAnalyzeFrame, 100);
});

videoEl.addEventListener('seeked', function(){
  if(videoEl.paused) setTimeout(triggerAnalyzeFrame, 100);
});

</script>
<div id="progress-overlay">
  <div id="progress-panel">
    <h3>&#9881; 批量导出中...</h3>
    <div class="stage-row stage-active" id="st0">
      <div class="stage-label" id="st0-label">&#128247; 提取帧</div>
      <div class="stage-bar-bg"><div class="stage-bar-fill" id="st0-fill" style="width:0%"></div></div>
      <div class="stage-pct" id="st0-pct">0%</div>
    </div>
    <div class="stage-row stage-pending" id="st1">
      <div class="stage-label" id="st1-label">&#128066; 提取人脸</div>
      <div class="stage-bar-bg"><div class="stage-bar-fill" id="st1-fill" style="width:0%"></div></div>
      <div class="stage-pct" id="st1-pct">-</div>
    </div>
    <div class="stage-row stage-pending" id="st2">
      <div class="stage-label" id="st2-label">&#127912; 换脸合成</div>
      <div class="stage-bar-bg"><div class="stage-bar-fill" id="st2-fill" style="width:0%"></div></div>
      <div class="stage-pct" id="st2-pct">-</div>
    </div>
    <div id="progress-time"></div>
    <button id="progress-cancel">取消</button>
  </div>
</div>

<!-- Step 0: Export confirmation dialog -->
<div id="export-dialog-overlay" class="export-dialog-overlay" style="display:none" onclick="if(event.target===this)this.style.display='none'">
  <div class="export-dialog">
    <h3>&#9881; 批量导出设置</h3>
    <div class="row"><label>起始时间</label><span id="exp-start-time">--</span></div>
    <div class="row"><label>结束时间</label><span id="exp-end-time">--</span></div>
    <div class="row"><label>总帧数</label><span id="exp-total-frames">--</span></div>
    <div class="row"><label>输出格式</label><select id="exp-format" onchange="document.getElementById('exp-jpeg-quality').disabled=(this.value!=='jpg')"><option value="jpg">JPEG</option><option value="png">PNG</option></select></div>
    <div class="row"><label>JPEG 质量</label><input type="number" id="exp-jpeg-quality" value="95" min="1" max="100"></div>
    <div class="row"><label>输出目录</label><input type="text" id="exp-output-dir" style="width:200px;font-size:11px"></div>
    <div class="row"><label>等待人脸筛选</label><input type="checkbox" id="exp-need-filter" style="width:auto"></div>
    <div class="btns">
      <button class="btn-cancel" onclick="document.getElementById('export-dialog-overlay').style.display='none'">取消</button>
      <button class="btn-confirm" onclick="onExportConfirm()">确认开始导出</button>
    </div>
  </div>
</div>

<!-- Step 2.5: Face filter waiting UI -->
<div id="waiting-ui-overlay" class="waiting-ui-overlay" style="display:none">
  <div class="waiting-ui">
    <h3>&#128066; 人脸筛选</h3>
    <p>已提取完成，请在目录中筛选人脸</p>
    <div class="dir-path">对齐目录: <span id="waiting-dir-path">--</span></div>
    <div class="btns">
      <button class="btn-open" onclick="fetch('/studio/open-directory',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:document.getElementById('waiting-dir-path').textContent})})">&#128193; 打开目录</button>
      <button class="btn-cluster" onclick="fetch('/studio/cluster-faces',{method:'POST',headers:{'Content-Type':'application/json'}}).then(function(r){return r.json();}).then(function(d){if(d.error)alert(d.error)})">&#128200; 聚类人脸</button>
      <button class="btn-organize" onclick="fetch('/studio/merge-aligned',{method:'POST',headers:{'Content-Type':'application/json'}}).then(function(r){return r.json();}).then(function(d){if(d.error)alert(d.error)})">&#128451; 整理文件夹</button>
      <button class="btn-continue" onclick="onContinueMerge()">&#9654; 继续合成</button>
      <button class="btn-cancel-wait" onclick="onCancelExport()">&#10005; 取消导出</button>
    </div>
  </div>
</div>
</body>
</html>"""
