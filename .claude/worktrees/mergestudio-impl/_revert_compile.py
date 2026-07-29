import sys

# ============ SAEHD ============
fp = r'C:\MySoftware\dfl-pytorch\DeepFaceLab-Torch-main\models\Model_SAEHD\Model_pytorch.py'
with open(fp, 'rb') as f:
    c = f.read()

# 1. Remove _forward_and_losses method
old = b'''    # --- compiled forward+loss (XLA) ---
    def _forward_and_losses(self, warped_src, target_src, target_srcm, target_srcm_em,
                            warped_dst, target_dst, target_dstm, target_dstm_em):
        """Combined forward + recon_losses, compiled as a single graph."""
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=self.use_bf16):
            if 'df' in self.archi_type:
                fw = self._forward_df(warped_src, warped_dst)
            else:
                fw = self._forward_liae(warped_src, warped_dst)

        if self.use_bf16:
            fw = {k: (v.float() if isinstance(v, torch.Tensor) else v) for k, v in fw.items()}

        src_loss_vec, dst_loss_vec, extra_style_loss, extra_masked_gan_loss = self._recon_losses(
            target_src, target_dst, target_srcm, target_dstm, target_srcm_em, target_dstm_em, fw,
        )

        G_loss = src_loss_vec.mean() + dst_loss_vec.mean() + extra_style_loss + extra_masked_gan_loss
        return G_loss, src_loss_vec, dst_loss_vec, fw

'''
if old in c:
    c = c.replace(old, b'', 1)
    print('SAEHD: removed _forward_and_losses')
else:
    print('SAEHD: _forward_and_losses NOT FOUND')
    # debug
    idx = c.find(b'_forward_and_losses')
    if idx >= 0: print(f'  found at {idx}, repr={repr(c[idx:idx+60])}')

# 2. Remove compile block from on_initialize
old = b'''        # torch.compile() combined forward+loss
        import torch
        try:
            self._fw_compiled = torch.compile(self._forward_and_losses, mode="default")
            io.log_info('torch.compile() enabled for SAEHD forward+loss')
        except Exception as e:
            self._fw_compiled = None
            io.log_info(f'torch.compile() failed: {e}, using eager mode')

'''
if old in c:
    c = c.replace(old, b'', 1)
    print('SAEHD: removed compile block from on_initialize')
else:
    print('SAEHD: compile block NOT FOUND')

# 3. Restore train_one_step to eager-only
old = b'''        if self._fw_compiled is not None:
            torch.compiler.cudagraph_mark_step_begin()
            try:
                G_loss, src_loss_vec, dst_loss_vec, fw = self._fw_compiled(
                    warped_src, target_src, target_srcm, target_srcm_em,
                    warped_dst, target_dst, target_dstm, target_dstm_em,
                )
            except Exception:
                self._fw_compiled = None
                io.log_info('torch.compile() failed at runtime, using eager mode')

        if self._fw_compiled is None:
'''
if old in c:
    c = c.replace(old, b'        ', 1)
    print('SAEHD: restored train_one_step to eager-only')
else:
    print('SAEHD: train_one_step dispatch NOT FOUND')

with open(fp, 'wb') as f:
    f.write(c)

# ============ AMP ============
fp2 = r'C:\MySoftware\dfl-pytorch\DeepFaceLab-Torch-main\models\Model_AMP\Model_pytorch.py'
with open(fp2, 'rb') as f:
    c2 = f.read()

# 1. Remove _forward_and_losses from AMP
# Find exact start/end using markers
mark_start = b'\t# --- compiled forward+loss (XLA) ---\r\n'
mark_mid  = b'def _forward_and_losses'
mark_end  = b'\t\treturn G_loss, src_loss_vec, dst_loss_vec, fw\r\n\r\n\t\t# --- losses / train ---'
idx_s = c2.find(mark_start)
if idx_s < 0:
    # try without (XLA)
    mark_start = b'\t# --- compiled forward+loss ---\r\n'
    idx_s = c2.find(mark_start)
idx_e = c2.find(mark_end, idx_s) if idx_s >= 0 else -1
if idx_s >= 0 and idx_e > idx_s:
    # remove from start of marker to before '# --- losses / train ---'
    c2 = c2[:idx_s] + b'\t\t' + c2[idx_e+len(mark_end)-2:]  # keep \t\t# --- losses...
    print('AMP: removed _forward_and_losses')
else:
    print(f'AMP: _forward_and_losses NOT FOUND ({idx_s=}, {idx_e=})')

# 2. Remove compile block from AMP on_initialize
old = b'\t# torch.compile() combined forward pass\r\n\t\timport torch\r\n\t\ttry:\r\n\t\t\tself._fw_compiled = torch.compile(self._forward_and_losses, mode="default")\r\n\t\t\tio.log_info("torch.compile() enabled for AMP forward pass")\r\n\t\texcept Exception as e:\r\n\t\t\tself._fw_compiled = None\r\n\t\t\tio.log_info(f"torch.compile() failed: {e}, using eager mode")\r\n\r\n'
if old in c2:
    c2 = c2.replace(old, b'', 1)
    print('AMP: removed compile block from on_initialize')
else:
    print('AMP: compile block NOT FOUND')
    # try the earlier version
    idx = c2.find(b'torch.compile() combined forward pass')
    if idx >= 0:
        chunk = c2[idx:idx+300]
        print(f'  Found at {idx}:')
        # show first 3 lines
        lines = chunk.split(b'\r\n')[:6]
        for li in lines:
            print(f'  {repr(li[:100])}')

# 3. Restore AMP train_one_step to eager-only
old = b'''\t\tif getattr(self, "_fw_compiled", None) is not None:
\t\t\ttorch.compiler.cudagraph_mark_step_begin()
\t\t\ttry:
\t\t\t\tG_loss, src_loss_vec, dst_loss_vec, fw = self._fw_compiled(
\t\t\t\t\twarped_src, target_src, target_srcm, target_srcm_em,
\t\t\t\t\twarped_dst, target_dst, target_dstm, target_dstm_em,
\t\t\t\t)
\t\t\texcept Exception:
\t\t\t\tself._fw_compiled = None
\t\t\t\tio.log_info("torch.compile() failed at runtime, using eager mode")

\t\tif getattr(self, "_fw_compiled", None) is None:
\t\t\tG_loss, src_loss_vec, dst_loss_vec, fw = self._forward_and_losses(
\t\t\t\twarped_src, target_src, target_srcm, target_srcm_em,
\t\t\t\twarped_dst, target_dst, target_dstm, target_dstm_em,
\t\t\t)
'''
if old in c2:
    c2 = c2.replace(old, b'\t\tG_loss, src_loss_vec, dst_loss_vec, fw = self._forward_and_losses(\n\t\t\twarped_src, target_src, target_srcm, target_srcm_em,\n\t\t\twarped_dst, target_dst, target_dstm, target_dstm_em,\n\t\t)\n', 1)
    print('AMP: restored train_one_step to eager-only')
else:
    print('AMP: train_one_step dispatch NOT FOUND')
    # try partial
    idx = c2.find(b'getattr(self, "_fw_compiled"')
    if idx >= 0:
        hunk = c2[idx:idx+400]
        print(f'  Found at {idx}:')
        lines = hunk.split(b'\r\n')[:12]
        for li in lines:
            print(f'  {repr(li[:100])}')

with open(fp2, 'wb') as f:
    f.write(c2)

print('\nDone')
