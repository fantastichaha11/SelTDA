#!/usr/bin/env bash
# RL teacher report focused on reward score & eval results.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 - <<'PY'
import json
import re
import statistics
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

root = Path("/home/phongnguyen/SelTDA")
log = root / "logs/train_vqg_rl_full.log"
epoch_log = root / "orchestration/state/round_1_epochs.jsonl"
audit_log = root / "orchestration/state/round_1_audit.jsonl"
weak_types_path = root / "orchestration/state/weak_types.json"
student_base_eval = root / "cache/evals/student_base/vqa_result.json"
round_state_glob = list((root / "orchestration/state").glob("round_*.json"))

import yaml
_cfg_path = root / "configs/rl_teacher_aokvqa.yaml"
steps_per_epoch = 200
if _cfg_path.exists():
    try:
        with open(_cfg_path) as _f:
            steps_per_epoch = int(
                yaml.safe_load(_f).get("grpo", {}).get("steps_per_epoch", 200)
            )
    except Exception:
        pass
epochs_per_round = 3
now = datetime.utcnow()

# --- training alive? ---
running = False
try:
    subprocess.check_output(["pgrep", "-f", "python train_vqg_rl.py"], stderr=subprocess.DEVNULL)
    running = True
except subprocess.CalledProcessError:
    pass

# --- parse GRPO log ---
text = log.read_text(errors="replace") if log.exists() else ""
group_size = 28

# Anchor current segment: last resume or auto-tune
segment_start = None
for line in text.splitlines():
    if "Auto-tuned images_per_step=" in line:
        m = re.search(r"group_size=(\d+)", line)
        if m:
            group_size = int(m.group(1))
        ts = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if ts:
            segment_start = datetime.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S")
    if "Resume: skip probe" in line or "Resumed policy from" in line:
        ts = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if ts:
            segment_start = datetime.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S")
        m = re.search(r"group=(\d+)", line)
        if m:
            group_size = int(m.group(1))

pat = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - __main__ - INFO - GRPO step (\d+) "
    r"mean_reward=([\d.]+)"
)
epoch_done_pat = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - __main__ - INFO - "
    r"Round (\d+) epoch (\d+): mean_reward=([\d.]+)"
)
audit_log_pat = re.compile(
    r"Audit epoch (\d+): judge=([\d.]+) \(itm=([\d.]+) vqa=([\d.]+)\) "
    r"reward=([\d.]+) hacking=(\w+) hrp=(\S+)"
)

all_grpo = []
epoch_done_ts = []
for line in text.splitlines():
    m = epoch_done_pat.search(line)
    if m:
        epoch_done_ts.append(
            datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        )
    m = pat.search(line)
    if m:
        all_grpo.append(
            {
                "ts": datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"),
                "step": int(m.group(2)),
                "reward": float(m.group(3)),
            }
        )

# Completed epochs from jsonl (used for epoch index + log cutoff)
full_epochs_pre = []
if epoch_log.exists():
    for line in epoch_log.read_text().strip().splitlines():
        try:
            e = json.loads(line)
            if e.get("steps") == steps_per_epoch:
                full_epochs_pre.append(e)
        except json.JSONDecodeError:
            pass

# Steps reset each epoch — anchor after last completed epoch and/or resume
cutoff_ts = None
if full_epochs_pre:
    te = len(full_epochs_pre) - 1
    for line in reversed(text.splitlines()):
        if f"Round 1 epoch {te}:" in line:
            ts = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
            if ts:
                cutoff_ts = datetime.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S")
            break
if segment_start:
    cutoff_ts = segment_start if cutoff_ts is None else max(cutoff_ts, segment_start)
if cutoff_ts:
    rows = [r for r in all_grpo if r["ts"] > cutoff_ts]
else:
    rows = all_grpo

step = rows[-1]["step"] if rows else 0
last = rows[-1] if rows else None
rewards = [r["reward"] for r in rows]

epoch_idx = len(full_epochs_pre)
step_in_epoch = step if rows else 0

# pace
pace = 2.4
if len(rows) >= 2:
    a, b = rows[-2], rows[-1]
    pace = max(0.1, (b["ts"] - a["ts"]).total_seconds() / 60 / max(1, b["step"] - a["step"]))

# hourly delta
cutoff = now - timedelta(hours=1)
hour = [r for r in rows if r["ts"] >= cutoff]

# epoch jsonl (completed epoch summaries)
epoch_rows = []
if epoch_log.exists():
    for line in epoch_log.read_text().strip().splitlines():
        try:
            epoch_rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass

full_epochs = [e for e in epoch_rows if e.get("steps") == steps_per_epoch]
smoke_epochs = [e for e in epoch_rows if e.get("steps", 0) < steps_per_epoch]

# --- audit jsonl (+ log fallback while audit runs) ---
audit_rows = []
if audit_log.exists():
    for line in audit_log.read_text().strip().splitlines():
        try:
            audit_rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
if not audit_rows:
    for m in audit_log_pat.finditer(text):
        audit_rows.append(
            {
                "epoch": int(m.group(1)),
                "mean_judge": float(m.group(2)),
                "mean_itm_judge": float(m.group(3)),
                "mean_vqa_match": float(m.group(4)),
                "audit_mean_reward": float(m.group(5)),
                "hacking": m.group(6) == "True",
                "hrp_action": m.group(7),
            }
        )

# --- eval: baseline student ---
baseline_acc = None
baseline_per_type = {}
if student_base_eval.exists():
    try:
        from vqa_eval_tools.vqa_eval import VQAEval

        results = json.loads(student_base_eval.read_text())
        ev = VQAEval(n=2)
        ev.evaluate(results, "dummy")
        baseline_acc = ev.accuracy.get("overall")
        baseline_per_type = ev.accuracy.get("perQuestionType", {})
    except Exception:
        try:
            data = json.loads(student_base_eval.read_text())
            if isinstance(data, dict) and "overall" in data:
                baseline_acc = data["overall"]
                baseline_per_type = data.get("perQuestionType", {})
        except Exception:
            pass

# weak types (from eval or default)
weak_types = []
if weak_types_path.exists():
    weak_types = json.loads(weak_types_path.read_text()).get("weak_types", [])

# round-level eval files
round_evals = []
for p in sorted(round_state_glob):
    try:
        round_evals.append((p.name, json.loads(p.read_text())))
    except Exception:
        pass

# --- running mean estimate for current epoch ---
# interpolate: each logged point = mean over ~10 steps
def epoch_running_mean(target_step):
    if not rows:
        return None
    pts = [(r["step"], r["reward"]) for r in rows if r["step"] <= target_step]
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0][1]
    total = 0.0
    prev_s, prev_r = pts[0]
    total += prev_r * min(10, prev_s)
    for s, r in pts[1:]:
        span = s - prev_s
        total += r * span
        prev_s, prev_r = s, r
    if target_step > prev_s:
        total += prev_r * (target_step - prev_s)
    return total / target_step

run_mean = epoch_running_mean(step) if step else None

# reward phases
early = [r["reward"] for r in rows if r["step"] <= 50]
mid = [r["reward"] for r in rows if 50 < r["step"] <= 120]
late = [r["reward"] for r in rows if r["step"] > 120]

print(f"# Báo cáo Reward & Eval — RL teacher GRPO")
print(
    f"**{now.strftime('%H:%M UTC %d/%m/%Y')}** · Round 1 · "
    f"Epoch {epoch_idx + 1}/{epochs_per_round} · Step {step_in_epoch}/{steps_per_epoch}"
)
print()

# === REWARD ===
print("## 1. Reward score (proxy GRPO)")
print()
print(
    "Reward = weighted sum of **type_match + ITM + grounding + learnability − repetition** "
    f"(weak types: {', '.join(weak_types) or 'N/A'})."
)
print()
if last:
    status = "🟢 training" if running else "🔴 stopped"
    print(f"**Trạng thái:** {status} · **Step log:** #{last['step']} @ {last['ts'].strftime('%H:%M UTC')} · **Batch mean:** **{last['reward']:.4f}**")
    if run_mean:
        print(f"**Running mean epoch {epoch_idx + 1} (ước lượng):** {run_mean:.4f}")
print()

print("| Thống kê (deca-log) | Giá trị |")
print("|---------------------|---------|")
if rewards:
    print(f"| Mean | {statistics.mean(rewards):.4f} |")
    print(f"| Min / Max | {min(rewards):.4f} / {max(rewards):.4f} |")
    if len(rewards) > 1:
        print(f"| Std | {statistics.stdev(rewards):.4f} |")
    print(f"| Latest (step {last['step']}) | {last['reward']:.4f} |")
    if len(rows) >= 2:
        dr = rows[-1]["reward"] - rows[0]["reward"]
        print(f"| Δ step {rows[0]['step']}→{rows[-1]['step']} | {dr:+.4f} |")
print()

if early or mid or late:
    print("**Theo giai đoạn (mean deca-log):**")
    print()
    print("| Giai đoạn | Steps | Mean reward |")
    print("|-----------|-------|-------------|")
    if early:
        print(f"| Khởi động | 1–50 | {statistics.mean(early):.4f} |")
    if mid:
        print(f"| Giữa | 51–120 | {statistics.mean(mid):.4f} |")
    if late:
        print(f"| Gần đây | 121+ | {statistics.mean(late):.4f} |")
    print()

if hour:
    print("**60 phút qua:**")
    print()
    h0, h1 = hour[0], hour[-1]
    print(f"- Steps logged: {h0['step']} → {h1['step']} ({h1['step'] - h0['step']:+d})")
    print(f"- Reward: {h0['reward']:.4f} → {h1['reward']:.4f} ({h1['reward'] - h0['reward']:+.4f})")
    print(f"- Tốc độ: ~{pace:.2f} phút/step")
    print()

print("**Toàn bộ curve (10-step log):**")
print()
print("| Step | Reward | Δ prev | Thời gian |")
print("|------|--------|--------|-----------|")
prev_r = None
for r in rows:
    delta = f"{r['reward'] - prev_r:+.4f}" if prev_r is not None else "—"
    print(f"| {r['step']} | {r['reward']:.4f} | {delta} | {r['ts'].strftime('%H:%M')} |")
    prev_r = r["reward"]
print()

# === EVAL ===
print("## 2. Eval results")
print()
print("Hai loại metric khác nhau:")
print("- **Reward (mục 1):** proxy online mỗi GRPO step — teacher đang tối ưu cái này.")
print("- **Eval (mục này):** metric *thật* — baseline student, judge audit, val accuracy sau train student.")
print()

print("### 2a. Baseline student (frozen, trước RL)")
print()
if baseline_acc is not None:
    print(f"| Metric | Giá trị |")
    print(f"|--------|---------|")
    print(f"| Val overall accuracy | **{baseline_acc:.2f}%** |")
    if baseline_per_type:
        print()
        print("**Theo question type (baseline):**")
        print()
        print("| Type | Accuracy |")
        print("|------|----------|")
        for t, v in sorted(baseline_per_type.items(), key=lambda x: x[1]):
            mark = " ← weak" if t in weak_types else ""
            print(f"| {t} | {v:.2f}%{mark} |")
else:
    print("- ⚠ Chưa có `cache/evals/student_base/vqa_result.json` trên máy này.")
    print("- Weak types đang dùng default/config: **" + ", ".join(weak_types) + "**")
print()

print("### 2b. Epoch-level (GRPO proxy summary)")
print()
if full_epochs:
    has_judge = any("mean_judge" in e for e in full_epochs)
    if has_judge:
        print("| Epoch | mean_reward | group_size | kl_beta | mean_judge | hacking |")
        print("|-------|-------------|------------|---------|------------|---------|")
    else:
        print("| Epoch | mean_reward | group_size | kl_beta |")
        print("|-------|-------------|------------|---------|")
    for e in full_epochs:
        row = (
            f"| {e.get('epoch', '?')} | {e.get('mean_reward', 0):.4f} | "
            f"{e.get('group_size', '?')} | {e.get('kl_beta', '?')} |"
        )
        if has_judge:
            mj = e.get("mean_judge")
            hack = e.get("hacking")
            row += (
                f" {mj:.4f} |" if mj is not None else " — |"
            )
            row += f" {hack} |" if hack is not None else " — |"
        print(row)
else:
    est_epoch_mean = run_mean or (statistics.mean(rewards) if rewards else None)
    rem = steps_per_epoch - step_in_epoch
    eta_ep = timedelta(minutes=rem * pace)
    print(f"- Epoch {epoch_idx + 1} **chưa xong** ({step_in_epoch}/{steps_per_epoch} steps)")
    if est_epoch_mean:
        print(f"- Running mean reward (ước lượng): **{est_epoch_mean:.4f}**")
    print(f"- Dự kiến ghi jsonl sau epoch: ~{(now + eta_ep).strftime('%H:%M UTC')}")
    if smoke_epochs:
        print("- _(Chỉ có smoke-test entries cũ trong jsonl, bỏ qua)_")
print()

print("### 2c. Audit judge (epoch-level, độc lập reward)")
print()
print(
    "Judge = 0.5×ITM-Large + 0.5×VQA match (checkpoint độc lập, khác reward student). "
    "Nguồn: `orchestration/state/round_1_audit.jsonl`."
)
print()
if audit_rows:
    print("| Epoch | mean_judge | ITM | VQA | audit_reward | Δ judge | hacking | HRP | note |")
    print("|-------|------------|-----|-----|----------------|---------|---------|-----|------|")
    prev_j = None
    for a in sorted(audit_rows, key=lambda x: int(x.get("epoch", 0))):
        ep = a.get("epoch", "?")
        mj = a.get("mean_judge")
        itm = a.get("mean_itm_judge")
        vqa = a.get("mean_vqa_match")
        ar = a.get("audit_mean_reward", a.get("mean_reward"))
        dj = a.get("d_judge")
        if dj is None and prev_j is not None and mj is not None:
            dj = mj - prev_j
        hack = a.get("hacking", False)
        hrp = a.get("hrp_action", "—")
        note = ""
        if a.get("backfill"):
            note = "backfill"
        if a.get("approximate"):
            note = (note + ", approx e0").strip(", ") if note else "approx e0"
        dj_s = f"{dj:+.4f}" if dj is not None else "—"
        print(
            f"| {ep} | {mj:.4f} | {itm:.4f} | {vqa:.4f} | {ar:.4f} | {dj_s} | "
            f"{'⚠' if hack else 'ok'} | {hrp} | {note or '—'} |"
        )
        if mj is not None:
            prev_j = mj
    print()
    last_a = audit_rows[-1]
    if last_a.get("term_means"):
        print("**Reward terms (audit sample, mean):**")
        print()
        print("| Term | Mean |")
        print("|------|------|")
        for k, v in sorted(last_a["term_means"].items()):
            print(f"| {k} | {v:.4f} |")
        print()
    # reward vs judge gap on last epoch
    grpo_r = None
    for e in full_epochs:
        if e.get("epoch") == last_a.get("epoch"):
            grpo_r = e.get("mean_reward")
            break
    if grpo_r is not None and last_a.get("audit_mean_reward") is not None:
        gap = float(grpo_r) - float(last_a["audit_mean_reward"])
        print(
            f"- Epoch {last_a.get('epoch')}: GRPO mean_reward **{grpo_r:.4f}** vs "
            f"audit sample reward **{last_a['audit_mean_reward']:.4f}** (Δ {gap:+.4f})."
        )
        print(
            f"- Judge **{last_a.get('mean_judge', 0):.4f}** — metric đối chiếu hacking "
            f"(reward ↑ mà judge ↓ → HRP)."
        )
        print()
elif running and step_in_epoch >= steps_per_epoch - 5:
    print("- ⏳ Epoch sắp xong — audit judge sẽ chạy ngay sau epoch (~30–60 phút, CPU).")
    print()
else:
    n_done = len(audit_rows)
    n_full = len(full_epochs)
    if n_full > n_done:
        print(
            f"- ⚠ Đã xong **{n_full}** epoch GRPO nhưng chỉ có **{n_done}** audit "
            f"(epoch cũ trước khi bật audit)."
        )
    else:
        print("- **Chưa có** audit — chạy sau mỗi epoch khi GRPO xong (ITM-Large + VQA judge, ~200 mẫu).")
    print()

print("### 2d. Round-level eval (val accuracy student_r)")
print()
if round_evals:
    for name, data in round_evals:
        acc = data.get("acc") or data.get("accuracy")
        print(f"- `{name}`: accuracy = **{acc}**")
else:
    print("- **Chưa có** — chỉ chạy sau khi Round 1 GRPO xong → generate → train student → eval val.")
    rem_epochs = max(0, epochs_per_round - epoch_idx - 1)
    rem_steps = max(0, steps_per_epoch - step_in_epoch) + rem_epochs * steps_per_epoch
    print(f"- GRPO còn **{rem_steps}** steps (~{rem_steps * pace / 60:.1f}h) trước khi tới bước eval round.")
print()

# === INTERPRETATION ===
print("## 3. Đọc nhanh")
print()
if rewards and len(rewards) >= 3:
    trend = "tăng" if rewards[-1] > rewards[0] + 0.1 else ("giảm" if rewards[-1] < rewards[0] - 0.1 else "ổn định")
    print(
        f"- Reward proxy epoch {epoch_idx + 1} **{trend}** "
        f"(step {rows[0]['step']}→{last['step']}): {rewards[0]:.2f} → {rewards[-1]:.2f}."
    )
if audit_rows:
    hacked = [a for a in audit_rows if a.get("hacking")]
    if hacked:
        eps = ", ".join(str(a.get("epoch")) for a in hacked)
        print(f"- ⚠ Hacking flag epoch **{eps}** — xem HRP action trong mục 2c.")
    elif len(audit_rows) >= 2:
        j0, j1 = audit_rows[-2].get("mean_judge"), audit_rows[-1].get("mean_judge")
        if j0 is not None and j1 is not None:
            jt = "tăng" if j1 > j0 + 0.02 else ("giảm" if j1 < j0 - 0.02 else "ổn định")
            print(f"- Judge score **{jt}** ({j0:.4f} → {j1:.4f}) — đối chiếu reward proxy.")
if baseline_acc is not None:
    print(f"- Baseline student val: **{baseline_acc:.2f}%** — metric đối chiếu cuối round.")
else:
    print("- Chưa có baseline val acc local → không so sánh được reward vs accuracy thật.")
print("- Reward ↑ **không** đồng nghĩa val acc ↑ — cần round eval + audit judge xác nhận.")
print()
print("---")
print(
    "_Log: `logs/train_vqg_rl_full.log` · epochs: `orchestration/state/round_1_epochs.jsonl` · "
    "audit: `orchestration/state/round_1_audit.jsonl`_"
)
PY
