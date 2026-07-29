#!/usr/bin/env python3
"""
page_FlitGame - 样本筛选页面（模块）
提供后端函数和 HTML 模板，供 page_trainer 导入使用。
也可独立运行（python -m WebUI.page_flit_game）。
"""
import argparse
import base64
import json
import time
import cv2
import numpy as np
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

# ------------------------------------------------------------------
# 后端逻辑（公开接口，page_trainer 导入使用）
# ------------------------------------------------------------------

WORKSPACE = 'workspace'


def scan_aligned_dirs(workspace=None):
    """扫描 workspace 下所有 aligned 目录，返回 [{path, name, count}]"""
    ws = Path(workspace or WORKSPACE).resolve()
    dirs = []
    if not ws.exists():
        return dirs
    for p in ws.rglob('aligned'):
        if p.is_dir() and p.parent.parent == ws:
            files = sorted([f for f in p.iterdir()
                           if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
            dirs.append({
                'path': str(p),
                'name': str(p.relative_to(ws)),
                'count': len(files),
            })
    return dirs


def list_samples(dir_path, offset, count):
    """返回从 offset 开始的 count 个样本（base64 512x512 JPEG）。"""
    p = Path(dir_path)
    files = sorted([f for f in p.iterdir()
                    if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    total = len(files)
    selected = files[offset:offset + count]

    samples = []
    for f in selected:
        try:
            img_array = np.fromfile(str(f), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                continue
            if img.shape[0] != 512 or img.shape[1] != 512:
                img = cv2.resize(img, (512, 512))
            ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                samples.append({
                    'filename': f.name,
                    'path': str(f),
                    'data': base64.b64encode(jpg.tobytes()).decode(),
                })
        except Exception:
            continue

    return samples, total


def move_to_trash(file_paths):
    """移动文件到 aligned_trash 目录，返回成功移动的数量。"""
    moved = 0
    for fp_str in file_paths:
        fp = Path(fp_str)
        if not fp.exists():
            continue
        trash_dir = fp.parent.parent / (fp.parent.name + '_trash')
        trash_dir.mkdir(parents=True, exist_ok=True)
        dst = trash_dir / fp.name
        try:
            fp.rename(dst)
            moved += 1
        except Exception:
            pass
    return moved


def count_files(dir_path):
    """统计目录中图片文件总数。"""
    p = Path(dir_path)
    if not p.exists():
        return 0
    return len([f for f in p.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])


# ------------------------------------------------------------------
# HTML / CSS / JS 前端
# ------------------------------------------------------------------

HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DFL 样本筛选</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f0f13;color:#e0e0e0;font-family:'Segoe UI',system-ui,sans-serif;height:100vh;overflow:hidden;display:flex;flex-direction:column}
header{display:flex;align-items:center;gap:10px;padding:10px 20px;background:#16161e;border-bottom:1px solid #2a2a3a;flex-wrap:wrap;flex-shrink:0}
.back-btn{display:inline-flex;align-items:center;gap:4px;padding:6px 12px;background:#2a2a3a;border:1px solid #3a3a5a;border-radius:6px;color:#7eb8f7;cursor:pointer;font-size:13px;text-decoration:none;white-space:nowrap}
.back-btn:hover{background:#3a3a5a}
select,input[type=number]{padding:5px 10px;background:#1e1e2e;border:1px solid #3a3a5a;border-radius:6px;color:#e0e0e0;font-size:13px;outline:none}
select:focus,input:focus{border-color:#4a7ab7}
select{min-width:130px}
label{font-size:13px;color:#aaa;display:inline-flex;align-items:center;gap:5px;white-space:nowrap}
#gallery{display:flex;flex-wrap:wrap;gap:14px;padding:20px 20px 256px;justify-content:center;flex:1;min-height:0;overflow-y:auto;align-content:flex-start}
.img-card{display:flex;flex-direction:column;align-items:center;cursor:pointer;position:relative;border-radius:8px;overflow:hidden;border:2px solid transparent;transition:border-color .15s;background:#16161e}
.img-card:hover{border-color:#4a7ab7}
.img-card.marked{border-color:#f04747}
.img-card{max-width:calc(100vw - 40px)}
.img-card img{width:512px;max-width:100%;height:auto;aspect-ratio:1;object-fit:cover;display:block;border-radius:6px 6px 0 0;user-select:none;-webkit-user-drag:none}
.img-card .overlay{position:absolute;top:0;left:0;width:512px;max-width:100%;height:100%;background:rgba(0,0,0,.6);display:none;align-items:center;justify-content:center;border-radius:6px 6px 0 0;pointer-events:none}
.img-card.marked .overlay{display:flex}
.img-card .overlay span{color:#ff6b6b;font-size:15px;font-weight:600;text-shadow:0 2px 8px rgba(0,0,0,.8);text-align:center;padding:8px}
.img-card .label{font-size:12px;color:#888;padding:6px 8px 8px;text-align:center;word-break:break-all;max-width:100%;line-height:1.3}
#footer{position:fixed;bottom:0;left:0;right:0;display:flex;flex-direction:column;align-items:stretch;padding:10px 20px 12px;background:#16161e;border-top:1px solid #2a2a3a;z-index:100;gap:8px}
#stats{font-size:13px;color:#aaa;display:flex;gap:18px;flex-wrap:wrap;justify-content:center}
#stats b{color:#e0e0e0}
#settle-btn{width:100%;display:inline-flex;align-items:center;justify-content:center;gap:6px;padding:8px 24px;background:#1e3a2e;border:1px solid #3a7a5a;border-radius:8px;color:#7ef7a0;cursor:pointer;font-size:15px;font-weight:600;transition:all .15s}
#settle-btn:hover{background:#2a5a3e}
#settle-btn:disabled{opacity:.35;cursor:default}
#jump-input{width:72px}
.empty-msg{color:#555;padding:60px 20px;font-size:14px;text-align:center;width:100%}
@media(max-width:1100px){.img-card img{width:360px;height:360px}.img-card .overlay{width:360px;height:360px}.img-card .label{max-width:360px}}
@media(max-width:800px){.img-card img{width:280px;height:280px}.img-card .overlay{width:280px;height:280px}.img-card .label{max-width:280px}header{padding:8px 12px;gap:6px}#footer{padding:10px 12px;flex-wrap:wrap;gap:8px}#stats{gap:10px}select{min-width:100px}}
/* password modal */
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:100;display:none;align-items:center;justify-content:center}
.modal-overlay.open{display:flex}
.modal-box{background:#16161e;border:1px solid #3a3a5a;border-radius:12px;padding:24px 32px;width:320px;text-align:center}
.modal-box h2{font-size:16px;color:#e0e0e0;margin-bottom:6px}
.modal-box p{font-size:12px;color:#888;margin-bottom:16px}
.modal-box input{width:100%;padding:8px 12px;background:#1e1e2e;border:1px solid #3a3a5a;border-radius:6px;color:#e0e0e0;font-size:14px;text-align:center}
.modal-box input:focus{outline:none;border-color:#4a7ab7}
.modal-box .modal-actions{display:flex;gap:8px;margin-top:12px}
.modal-box .modal-actions button{flex:1;padding:8px;border-radius:6px;cursor:pointer;font-size:13px;font-weight:600}
.modal-box .modal-err{font-size:12px;color:#f77;margin-top:8px;display:none}
</style>
</head>
<body>
<!-- Password modal (for settle action) -->
<div class="modal-overlay" id="pwd-modal">
  <div class="modal-box">
    <h2 id="pwd-modal-title">&#128274; 需要密码确认</h2>
    <p id="pwd-modal-desc">结算操作需要输入密码</p>
    <input id="pwd-input" type="password" placeholder="请输入密码" autocomplete="off">
    <div class="modal-err" id="pwd-err">密码错误，请重试</div>
    <div class="modal-actions">
      <button class="btn-reset" id="pwd-cancel-btn" onclick="closePwdModal()">取消</button>
      <button class="btn-apply" id="pwd-confirm-btn" onclick="confirmPwd()">确认</button>
    </div>
  </div>
</div>
<header>
  <a href="/Trainer" class="back-btn">&larr; 返回训练</a>
  <select id="dir-select"><option value="">-- 选择目录 --</option></select>
  <label>每批
    <select id="count-select">
      <option value="2">2</option>
      <option value="4" selected>4</option>
      <option value="6">6</option>
      <option value="8">8</option>
    </select>
  </label>
  <label>跳至 <input type="number" id="jump-input" value="0" min="0"></label>
  <button class="back-btn" id="jump-btn">跳转</button>
</header>
<div id="gallery"><div class="empty-msg">请选择目录</div></div>
<div id="footer">
  <div id="stats">
    <span>已处理: <b id="s-done">0</b></span>
    <span>剩余: <b id="s-remain">0</b></span>
    <span>已移动: <b id="s-trashed">0</b></span>
  </div>
  <button id="settle-btn" disabled>&#9654; 结算本批</button>
</div>
<script>
var _dir = '';
var _offset = 0;
var _count = 4;
var _total = 0;
var _trashed = 0;
var _marked = {};
var _currentFiles = [];

// ---- 目录列表 ----
fetch('/FlitGame/api/dirs').then(function(r){return r.json()}).then(function(dirs){
  var sel = document.getElementById('dir-select');
  dirs.forEach(function(d){
    var opt = document.createElement('option');
    opt.value = d.path;
    opt.textContent = d.name + ' (' + d.count + ')';
    sel.appendChild(opt);
  });
});

// ---- 事件绑定 ----
var _authenticated = false;

document.getElementById('dir-select').addEventListener('change', function(){
  if(!_authenticated){
    // show password modal for dir selection
    _pendingDir = this.value;
    this.value = '';
    document.getElementById('pwd-modal-title').textContent = '🔒 需要密码';
    document.getElementById('pwd-modal-desc').textContent = '选择目录需要输入密码';
    document.getElementById('pwd-confirm-btn').style.background = '#2a5a3a';
    document.getElementById('pwd-confirm-btn').style.borderColor = '#3a7a5a';
    document.getElementById('pwd-confirm-btn').style.color = '#fff';
    document.getElementById('pwd-cancel-btn').style.display = 'block';
    document.getElementById('pwd-modal').classList.add('open');
    document.getElementById('pwd-input').value = '';
    document.getElementById('pwd-err').style.display = 'none';
    document.getElementById('pwd-input').focus();
    return;
  }
  _dir = this.value;
  _offset = 0;
  _marked = {};
  if(_dir) loadSamples();
  else document.getElementById('gallery').innerHTML = '<div class="empty-msg">请选择目录</div>';
});

document.getElementById('count-select').addEventListener('change', function(){
  _count = parseInt(this.value);
  _offset = 0;
  _marked = {};
  if(_dir) loadSamples();
});

document.getElementById('jump-btn').addEventListener('click', function(){
  var v = parseInt(document.getElementById('jump-input').value);
  if(v >= 0 && v !== _offset){
    _offset = v;
    _marked = {};
    if(_dir) loadSamples();
  }
});
document.getElementById('jump-input').addEventListener('keydown', function(e){
  if(e.key === 'Enter') document.getElementById('jump-btn').click();
});

// ---- 加载样本 ----
function loadSamples(){
  var gallery = document.getElementById('gallery');
  gallery.innerHTML = '<div class="empty-msg">加载中...</div>';
  document.getElementById('settle-btn').disabled = true;

  fetch('/FlitGame/api/samples?dir=' + encodeURIComponent(_dir) + '&offset=' + _offset + '&count=' + _count)
    .then(function(r){return r.json()})
    .then(function(data){
      _total = data.total;
      _currentFiles = data.samples;
      renderGallery();
      updateStats();
      document.getElementById('settle-btn').disabled = data.samples.length === 0;
    })
    .catch(function(){
      gallery.innerHTML = '<div class="empty-msg">加载失败，请重试</div>';
    });
}

// ---- 渲染图片网格 ----
function renderGallery(){
  var gallery = document.getElementById('gallery');
  gallery.innerHTML = '';
  if(!_currentFiles || !_currentFiles.length){
    gallery.innerHTML = '<div class="empty-msg">没有更多样本了</div>';
    return;
  }
  _currentFiles.forEach(function(item, idx){
    var card = document.createElement('div');
    card.className = 'img-card' + (_marked[item.path] ? ' marked' : '');
    card.dataset.idx = idx;

    var img = document.createElement('img');
    img.src = 'data:image/jpeg;base64,' + item.data;
    img.alt = item.filename;

    var overlay = document.createElement('div');
    overlay.className = 'overlay';
    var span = document.createElement('span');
    span.textContent = '⚠️ 即将排除这个样本';
    overlay.appendChild(span);

    var label = document.createElement('div');
    label.className = 'label';
    label.textContent = item.filename;

    card.appendChild(img);
    card.appendChild(overlay);
    card.appendChild(label);

    card.addEventListener('click', function(){
      var path = _currentFiles[parseInt(this.dataset.idx)].path;
      if(_marked[path]){
        delete _marked[path];
        this.classList.remove('marked');
      } else {
        _marked[path] = true;
        this.classList.add('marked');
      }
    });

    gallery.appendChild(card);
  });
}

// ---- 更新统计 ----
function updateStats(){
  var done = _offset;
  var remain = Math.max(0, _total - _offset);
  document.getElementById('s-done').textContent = done;
  document.getElementById('s-remain').textContent = remain;
  document.getElementById('s-trashed').textContent = _trashed;
}

// ---- 结算按钮 ----
// Password modal
var _pendingMarked = null;
var _pendingDir = null;
function openPwdModal() {
  document.getElementById('pwd-modal-title').textContent = '✅ 结算确认';
  document.getElementById('pwd-modal-desc').textContent = '输入密码将标记的图片移入回收站';
  document.getElementById('pwd-confirm-btn').style.background = '#2a5a3a';
  document.getElementById('pwd-confirm-btn').style.borderColor = '#3a7a5a';
  document.getElementById('pwd-confirm-btn').style.color = '#fff';
  document.getElementById('pwd-cancel-btn').style.display = 'block';
  document.getElementById('pwd-modal').classList.add('open');
  document.getElementById('pwd-input').value = '';
  document.getElementById('pwd-err').style.display = 'none';
  document.getElementById('pwd-input').focus();
}
function closePwdModal() {
  document.getElementById('pwd-modal').classList.remove('open');
  document.getElementById('pwd-confirm-btn').style.background = '';
  document.getElementById('pwd-confirm-btn').style.borderColor = '';
  document.getElementById('pwd-confirm-btn').style.color = '';
  _pendingMarked = null;
  _pendingDir = null;
}
document.getElementById('pwd-input').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') confirmPwd();
});
async function confirmPwd() {
  var pwd = document.getElementById('pwd-input').value;
  var err = document.getElementById('pwd-err');
  try {
    var r = await fetch('/check-password', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({password: pwd}),
    });
    var res = await r.json();
    if (res.ok) {
      // save before closePwdModal nulls them
      var marked = _pendingMarked;
      var dir = _pendingDir;
      closePwdModal();
      if (dir !== null) {
        _authenticated = true;
        _pendingDir = null;
        document.getElementById('dir-select').value = dir;
        _dir = dir;
        _offset = 0;
        _marked = {};
        if(_dir) loadSamples();
        else document.getElementById('gallery').innerHTML = '<div class="empty-msg">请选择目录</div>';
      } else if (marked) {
        doSettle(marked, pwd);
      }
    } else {
      err.style.display = 'block';
    }
  } catch(e) {
    err.textContent = '请求失败: ' + e.message;
    err.style.display = 'block';
  }
}
function doSettle(markedPaths, pwd) {
  var btn = document.getElementById('settle-btn');
  btn.disabled = true;
  btn.textContent = '⏳ 处理中...';
  fetch('/FlitGame/api/trash', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({paths: markedPaths, password: pwd})
  })
  .then(function(r){return r.json()})
  .then(function(result){
    _trashed += result.moved;
    _offset += _count;
    _marked = {};
    loadSamples();
    btn.textContent = '▶ 结算本批';
  })
  .catch(function(){
    btn.disabled = false;
    btn.textContent = '▶ 结算本批';
  });
}

document.getElementById('settle-btn').addEventListener('click', function(){
  var markedPaths = Object.keys(_marked);
  if(markedPaths.length === 0){
    _offset += _count;
    _marked = {};
    loadSamples();
    return;
  }
  if (_authenticated) {
    doSettle(markedPaths);
    return;
  }
  _pendingMarked = markedPaths;
  openPwdModal();
});
</script>
</body>
</html>"""


# ------------------------------------------------------------------
# 独立 HTTP 服务（仅 __main__ 使用）
# ------------------------------------------------------------------

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class _Handler(BaseHTTPRequestHandler):
    """独立模式下的 HTTP handler（集成模式下由 page_trainer 接管）"""
    def log_message(self, *a): pass

    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == '/api/dirs':
            self._json(scan_aligned_dirs())
        elif path == '/api/samples':
            dir_path = params.get('dir', [None])[0]
            if not dir_path:
                self._json({'samples': [], 'total': 0})
                return
            offset = int(params.get('offset', ['0'])[0])
            count = int(params.get('count', ['4'])[0])
            samples, total = list_samples(dir_path, offset, count)
            self._json({'samples': samples, 'total': total})
        elif path == '/api/stats':
            dir_path = params.get('dir', [None])[0]
            if not dir_path:
                self._json({'total': 0})
                return
            total = count_files(dir_path)
            trash_dir = Path(dir_path).parent / (Path(dir_path).name + '_trash')
            trashed = count_files(str(trash_dir)) if trash_dir.exists() else 0
            self._json({'total': total, 'trashed': trashed})
        else:
            body = HTML.encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_POST(self):
        if self.path == '/api/trash':
            try:
                content_len = int(self.headers.get('Content-Length', 0))
                raw = self.rfile.read(content_len)
                data = json.loads(raw)
                moved = move_to_trash(data.get('paths', []))
                self._json({'moved': moved})
            except Exception as e:
                self._json({'error': str(e)}, 400)
        else:
            self._json({'error': 'not found'}, 404)


def start_standalone_server(port=6790):
    """启动独立 HTTP 服务器（自测用）。"""
    ip = '127.0.0.1'
    try:
        import socket
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        pass
    print(f'[FlitGame] 样本筛选: http://{ip}:{port}')
    server = ThreadingHTTPServer(('0.0.0.0', port), _Handler)
    import threading
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DFL 样本筛选页面')
    parser.add_argument('--workspace', default='workspace',
                        help='workspace 目录路径')
    parser.add_argument('--port', type=int, default=6790,
                        help='HTTP 端口 (默认 6790)')
    args = parser.parse_args()
    WORKSPACE = args.workspace

    server = start_standalone_server(port=args.port)
    print(f'[FlitGame] 按 Ctrl+C 停止')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
        print('[FlitGame] 已停止')
