#!/usr/bin/env python3
"""
page_settings - 运行时参数调整页面（模块）
提供 HTML 模板，供 page_trainer 导入使用。
"""
HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DFL 参数设置</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f0f13;color:#e0e0e0;font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh}
header{display:flex;align-items:center;gap:12px;padding:12px 20px;background:#16161e;border-bottom:1px solid #2a2a3a}
header h1{font-size:16px;font-weight:600;color:#7eb8f7}
.back-btn{padding:5px 12px;background:#2a2a3a;border:1px solid #3a3a5a;border-radius:6px;color:#aaa;cursor:pointer;font-size:13px;text-decoration:none}
.back-btn:hover{background:#3a3a5a;color:#e0e0e0}
main{padding:20px;display:flex;flex-direction:column;gap:16px;max-width:720px;margin:0 auto}
.card{background:#16161e;border:1px solid #2a2a3a;border-radius:10px;padding:0}
.card h2{font-size:13px;font-weight:600;color:#888;text-transform:uppercase;letter-spacing:.8px;padding:14px 16px 0;margin-bottom:4px}
.card .note{font-size:11px;color:#555;padding:2px 16px 8px}
.row{display:flex;align-items:center;justify-content:space-between;padding:9px 16px;border-bottom:1px solid #1e1e2e;min-height:36px}
.row:last-child{border-bottom:none}
.row .lbl{font-size:13px;color:#ccc;user-select:none}
.row .val{font-size:12px;color:#888;font-family:monospace;background:#1a1a26;padding:3px 8px;border-radius:4px}
.row .ctrl{display:flex;align-items:center;gap:6px;flex-shrink:0}
.row input[type=text],.row input[type=number],.row select{background:#1e1e2e;border:1px solid #3a3a5a;border-radius:5px;padding:5px 8px;color:#e0e0e0;font-size:12px;font-family:monospace;width:80px;text-align:center}
.row input:focus,.row select:focus{outline:none;border-color:#4a7ab7}
.row select{width:auto;min-width:80px;cursor:pointer}
.row input[type=number]{width:80px}
.toggle{position:relative;width:44px;height:24px;flex-shrink:0}
.toggle input{opacity:0;width:0;height:0}
.toggle .slider{position:absolute;cursor:pointer;inset:0;background:#2a2a3a;border-radius:12px;transition:background .2s}
.toggle .slider::before{content:'';position:absolute;left:3px;top:3px;width:18px;height:18px;border-radius:50%;background:#666;transition:transform .2s,background .2s}
.toggle input:checked+.slider{background:#2a5a7a}
.toggle input:checked+.slider::before{transform:translateX(20px);background:#7eb8f7}
.range-ctrl{display:flex;align-items:center;gap:4px}
.range-ctrl input{width:60px}
.range-ctrl .sep{color:#444;font-size:11px}
.actions{display:flex;gap:10px;justify-content:flex-end;margin-top:6px;padding:0 16px 14px}
.actions button{padding:8px 20px;border-radius:6px;cursor:pointer;font-size:13px;font-weight:600}
.btn-apply{background:#2a5a7a;border:1px solid #3a7ab7;color:#fff}
.btn-apply:hover{background:#3a6a8a}
.btn-reset{background:#2a2a3a;border:1px solid #3a3a5a;color:#aaa}
.btn-reset:hover{background:#3a3a5a;color:#e0e0e0}
.btn-apply:disabled,.btn-reset:disabled{opacity:.4;cursor:default}
#toast{position:fixed;bottom:30px;left:50%;transform:translateX(-50%);padding:10px 24px;border-radius:8px;font-size:13px;opacity:0;transition:opacity .3s;pointer-events:none;z-index:100}
#toast.show{opacity:1}
#toast.ok{background:#2a5a3a;color:#fff}
#toast.err{background:#5a2a2a;color:#fff}

/* modal overlay */
.modal-overlay{position:fixed;inset:0;background:rgba(0,0,0,.75);z-index:50;display:none;align-items:center;justify-content:center}
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
<header>
  <a class="back-btn" href="/Trainer">&larr; 返回</a>
  <h1>运行时参数调整</h1>
</header>

<!-- Password modal (hidden until Apply clicked) -->
<div class="modal-overlay" id="pwd-modal">
  <div class="modal-box">
    <h2>&#128274; 需要密码确认</h2>
    <p>修改参数需要输入密码</p>
    <input id="pwd-input" type="password" placeholder="请输入密码" autocomplete="off">
    <div class="modal-err" id="pwd-err">密码错误，请重试</div>
    <div class="modal-actions">
      <button class="btn-reset" onclick="closePwdModal()">取消</button>
      <button class="btn-apply" onclick="confirmPassword()">确认</button>
    </div>
  </div>
</div>

<main>
  <div id="loading" style="color:#888;font-size:13px;padding:40px;text-align:center">正在加载设置…</div>
  <div id="content" style="display:none">
    <div class="card">
      <h2>模型架构</h2>
      <div class="note">仅作展示，不可修改</div>
      <div id="arch-fields"></div>
    </div>
    <div class="card">
      <h2>训练选项</h2>
      <div class="note">仅作展示，不可修改</div>
      <div id="train-fields"></div>
    </div>
    <div class="card">
      <h2>样本增强</h2>
      <div class="note">可动态调整</div>
      <div id="sp-fields"></div>
    </div>
    <div class="card">
      <h2>输出配置</h2>
      <div class="note">可动态调整</div>
      <div id="oc-fields"></div>
    </div>
    <div class="card">
      <h2>训练参数</h2>
      <div class="note">可动态调整</div>
      <div id="tr-fields"></div>
    </div>
    <div class="card">
      <h2>加载器</h2>
      <div class="note">可动态调整</div>
      <div id="ld-fields"></div>
    </div>
    <div class="card">
      <h2>保存与备份</h2>
      <div class="note">可动态调整</div>
      <div id="sv-fields"></div>
    </div>
    <div class="actions">
      <button class="btn-reset" onclick="loadAll()">重置表单</button>
      <button class="btn-apply" id="btn-apply" onclick="onApply()">应用参数</button>
    </div>
    <div id="last-refresh" style="text-align:center;font-size:11px;color:#444;padding:0 0 14px">等待加载...</div>
  </div>
</main>
<div id="toast"></div>

<script>
// ========== Field Definitions ==========
const ARCH_FIELDS = [
  ['resolution', '分辨率', v => v + 'px'],
  ['face_type', '人脸类型', null],
  ['archi', 'AE 架构', null],
  ['ae_dims', 'AE 维度', null],
  ['e_dims', '编码器维度', null],
  ['d_dims', '解码器维度', null],
  ['d_mask_dims', '解码器 Mask 维度', null],
];

const TRAIN_FIELDS = [
  ['adabelief', 'AdaBelief 优化器', v => v ? '&#10003;' : '&#10007;'],
  ['clipgrad', '梯度裁剪', v => v ? '&#10003;' : '&#10007;'],
  ['lr_dropout', '学习率 Dropout', null],
  ['gradient_checkpointing', '梯度检查点', v => v ? '&#10003;' : '&#10007;'],
  ['use_bf16', 'BF16 混合精度', v => v ? '&#10003;' : '&#10007;'],
  ['models_opt_on_gpu', '优化器在 GPU', v => v ? '&#10003;' : '&#10007;'],
  ['gan_power', 'GAN 强度', v => parseFloat(v).toFixed(4)],
  ['gan_patch_size', 'GAN 块大小', null],
  ['gan_dims', 'GAN 维度', null],
  ['true_face_power', 'True Face 强度', v => parseFloat(v).toFixed(4)],
  ['use_fast_generator', '快速生成器', v => v ? '&#10003;' : '&#10007;'],
  ['pretrain', '预训练模式', v => v ? '&#10003;' : '&#10007;'],
];

const TR_FIELDS = [
  ['tr', 'face_style_power', 'float', '人脸风格强度'],
  ['tr', 'bg_style_power', 'float', '背景风格强度'],
  ['tr', 'masked_training', 'toggle', '遮罩区域训练'],
  ['tr', 'blur_out_mask', 'toggle', '羽化遮罩外围'],
  ['tr', 'eyes_mouth_prio', 'toggle', '嘴眼优先'],
  ['tr', 'uniform_yaw', 'toggle', 'Yaw 均匀分布'],
];

const SP_FIELDS = [
  ['sp', 'random_flip', 'toggle', '随机翻转'],
  ['sp', 'rotation_range', 'range', '旋转范围'],
  ['sp', 'scale_range', 'range', '缩放范围'],
  ['sp', 'tx_range', 'range', '水平位移'],
  ['sp', 'ty_range', 'range', '垂直位移'],
];

const OC_FIELDS = [
  ['oc', 'warp', 'toggle', '扭曲'],
  ['oc', 'transform', 'toggle', '变换'],
  ['oc', 'ct_mode', 'select', '色彩迁移', {none:'无', rct:'RCT', lct:'LCT', mkl:'MKL', idt:'IDT', sot:'SOT', sot_s:'SOT-S'}],
  ['oc', 'random_hsv_shift_amount', 'float', 'HSV 偏移'],
];

const LD_FIELDS = [
  ['ld', 'batch_size', 'int', 'Batch Size'],
];

const SV_FIELDS = [
  ['sv', 'crash_threshold', 'float', '崩溃阈值'],
  ['sv', 'backup_interval', 'int', '备份间隔（迭代数）'],
  ['sv', 'max_backups', 'int', '最大备份数'],
];

// ========== State ==========
let _pendingPayload = null;

// ========== Password Modal ==========
function openPwdModal() {
  document.getElementById('pwd-modal').classList.add('open');
  document.getElementById('pwd-input').value = '';
  document.getElementById('pwd-err').style.display = 'none';
  document.getElementById('pwd-input').focus();
}
function closePwdModal() {
  document.getElementById('pwd-modal').classList.remove('open');
  _pendingPayload = null;
}
document.getElementById('pwd-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') confirmPassword();
});

async function confirmPassword() {
  const pwd = document.getElementById('pwd-input').value;
  const err = document.getElementById('pwd-err');
  try {
    const r = await fetch('/check-password', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({password: pwd}),
    });
    const res = await r.json();
    if (res.ok) {
      const payload = _pendingPayload;
      closePwdModal();
      await doApply(payload, pwd);
    } else {
      err.style.display = 'block';
    }
  } catch(e) {
    err.textContent = '请求失败: ' + e.message;
    err.style.display = 'block';
  }
}

// ========== Build ==========
function buildReadonlyRow(label, valueHtml) {
  const row = document.createElement('div');
  row.className = 'row';
  const lbl = document.createElement('span');
  lbl.className = 'lbl';
  lbl.textContent = label;
  const val = document.createElement('span');
  val.className = 'val';
  val.innerHTML = valueHtml;
  row.appendChild(lbl);
  row.appendChild(val);
  return row;
}

function buildField(group, key, type, label, opts) {
  const row = document.createElement('div');
  row.className = 'row';
  const lbl = document.createElement('span');
  lbl.className = 'lbl';
  lbl.textContent = label;
  row.appendChild(lbl);
  const ctrl = document.createElement('div');
  ctrl.className = 'ctrl';
  if (type === 'toggle') {
    const le = document.createElement('label');
    le.className = 'toggle';
    const cb = document.createElement('input');
    cb.type = 'checkbox'; cb.dataset.group = group; cb.dataset.fieldKey = key;
    const sp = document.createElement('span');
    sp.className = 'slider';
    le.appendChild(cb); le.appendChild(sp);
    ctrl.appendChild(le);
  } else if (type === 'select') {
    const sel = document.createElement('select');
    sel.dataset.group = group; sel.dataset.fieldKey = key;
    for (const [val, txt] of Object.entries(opts)) {
      const opt = document.createElement('option');
      opt.value = val; opt.textContent = txt;
      sel.appendChild(opt);
    }
    ctrl.appendChild(sel);
  } else if (type === 'range') {
    const rc = document.createElement('div');
    rc.className = 'range-ctrl';
    const i0 = document.createElement('input'); i0.type = 'text';
    i0.dataset.group = group; i0.dataset.fieldKey = key; i0.dataset.rangeIdx = '0'; i0.placeholder = 'min';
    const sp = document.createElement('span'); sp.className = 'sep'; sp.textContent = '~';
    const i1 = document.createElement('input'); i1.type = 'text';
    i1.dataset.group = group; i1.dataset.fieldKey = key; i1.dataset.rangeIdx = '1'; i1.placeholder = 'max';
    rc.appendChild(i0); rc.appendChild(sp); rc.appendChild(i1);
    ctrl.appendChild(rc);
  } else {
    const inp = document.createElement('input'); inp.type = 'text';
    inp.dataset.group = group; inp.dataset.fieldKey = key; inp.placeholder = key;
    ctrl.appendChild(inp);
  }
  row.appendChild(ctrl);
  return row;
}

function getVal(el) {
  if (el.type === 'checkbox') return el.checked;
  if (el.tagName === 'SELECT') return el.value;
  const raw = el.value.trim();
  const n = Number(raw);
  if (!isNaN(n) && raw !== '') return n;
  return raw;
}

function setVal(el, val) {
  if (el.type === 'checkbox') { el.checked = !!val; return; }
  if (el.tagName === 'SELECT') { el.value = String(val); return; }
  if (typeof val === 'number') { el.value = String(val); return; }
  el.value = val != null ? String(val) : '';
}

function formatVal(v, fmt) {
  if (fmt) return fmt(v);
  if (v === true) return '&#10003;';
  if (v === false) return '&#10007;';
  if (v == null) return '<span style="color:#555">&#8212;</span>';
  return String(v);
}

// ========== Data Loading ==========
async function loadAll() {
  document.getElementById('loading').style.display = 'block';
  document.getElementById('content').style.display = 'none';
  try {
    const [modelR, genR] = await Promise.all([
      fetch('/current-model-options'),
      fetch('/current-settings'),
    ]);
    const modelOpts = await modelR.json();
    const genSettings = await genR.json();
    renderAll(modelOpts, genSettings);
    document.getElementById('loading').style.display = 'none';
    document.getElementById('content').style.display = 'block';
    var _lr=document.getElementById('last-refresh');if(_lr)_lr.textContent='最后刷新: '+new Date().toLocaleTimeString();
  } catch(e) {
    document.getElementById('loading').textContent = '加载失败: ' + e.message;
  }
}

function renderAll(modelOpts, genSettings) {
  document.getElementById('arch-fields').innerHTML = '';
  ARCH_FIELDS.forEach(([key, label, fmt]) => {
    document.getElementById('arch-fields').appendChild(buildReadonlyRow(label, formatVal(modelOpts[key], fmt)));
  });
  document.getElementById('train-fields').innerHTML = '';
  TRAIN_FIELDS.forEach(([key, label, fmt]) => {
    document.getElementById('train-fields').appendChild(buildReadonlyRow(label, formatVal(modelOpts[key], fmt)));
  });
  populateEditable('sp-fields', 'sp', SP_FIELDS, genSettings.sample_process_options || {});
  populateEditable('oc-fields', 'oc', OC_FIELDS, genSettings.output_sample_types || {});
  populateEditable('tr-fields', 'tr', TR_FIELDS, modelOpts);
  populateEditable('ld-fields', 'ld', LD_FIELDS, genSettings.loader || {});
  populateEditable('sv-fields', 'sv', SV_FIELDS, modelOpts);
}

function populateEditable(containerId, group, fieldDefs, data) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  fieldDefs.forEach(([g, key, type, label, opts]) => {
    if (g !== group) return;
    const row = buildField(g, key, type, label, opts);
    container.appendChild(row);
    const els = container.querySelectorAll(`[data-group="${group}"][data-field-key="${key}"]`);
    els.forEach(el => {
      const ridx = el.dataset.rangeIdx;
      if (ridx !== undefined) {
        const v = data[key];
        if (Array.isArray(v)) setVal(el, v[parseInt(ridx)]);
      } else {
        setVal(el, data[key]);
      }
    });
  });
}

// ========== Collect ==========
function collectEditable() {
  const result = {};
  const rangeBuf = {};
  document.querySelectorAll('[data-group]').forEach(el => {
    const group = el.dataset.group;
    const key = el.dataset.fieldKey;
    const ridx = el.dataset.rangeIdx;
    if (!group || !key) return;
    if (!result[group]) result[group] = {};
    if (ridx !== undefined) {
      const bk = group + '\t' + key;
      if (!rangeBuf[bk]) rangeBuf[bk] = [];
      rangeBuf[bk][parseInt(ridx)] = getVal(el);
    } else {
      result[group][key] = getVal(el);
    }
  });
  for (const [k, arr] of Object.entries(rangeBuf)) {
    const sep = k.indexOf('\t');
    result[k.slice(0, sep)][k.slice(sep + 1)] = arr;
  }
  return result;
}

// ========== Apply ==========
function onApply() {
  const collected = collectEditable();
  const genPayload = {};
  if (collected.sp) genPayload.sample_process_options = collected.sp;
  if (collected.oc) genPayload.output_sample_types = collected.oc;
  if (collected.ld) genPayload.loader = collected.ld;
  const modelPayload = {};
  if (collected.tr) Object.assign(modelPayload, collected.tr);
  if (collected.sv) Object.assign(modelPayload, collected.sv);
  _pendingPayload = {genPayload, modelPayload};
  openPwdModal();
}

async function doApply(payload, pwd) {
  const {genPayload, modelPayload} = payload;
  const btn = document.getElementById('btn-apply');
  btn.disabled = true;
  btn.textContent = '应用中...';
  try {
    let ok = true;
    if (Object.keys(genPayload).length > 0) {
      const r1 = await fetch('/update-settings', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(Object.assign({}, genPayload, {password: pwd})),
      });
      if (!(await r1.json()).ok) ok = false;
    }
    if (Object.keys(modelPayload).length > 0) {
      const r2 = await fetch('/update-model-options', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(Object.assign({}, modelPayload, {password: pwd})),
      });
      if (!(await r2.json()).ok) ok = false;
    }
    showToast(ok ? '参数已应用，将在下一次迭代生效' : '部分参数应用失败', ok ? 'ok' : 'err');
  } catch(e) {
    showToast('请求失败: ' + e.message, 'err');
  } finally {
    btn.disabled = false;
    btn.textContent = '应用参数';
    _pendingPayload = null;
  }
}

function showToast(msg, type) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = type;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3000);
}

loadAll();
</script>
</body>
</html>"""
