# -*- coding: utf-8 -*-
"""
zmws.py — 造梦无双（4399 网页版）日常自动化辅助脚本
====================================================
通过本机 Kimi WebBridge daemon (HTTP http://127.0.0.1:10086/command) 控制
Edge 浏览器，对 canvas 游戏画面做「截图 + 像素色判断 + 坐标点击」。

坐标系：
  - 游戏在页面 iframe 里，元素位置用相对 iframe 的比例 (fx, fy) 表示。
  - 在线：evaluate 取 iframe CSS 矩形 r 和 dpr；
    点击 CSS 坐标 = (r.x + fx*r.width, r.y + fy*r.height)；
    截图像素     = ((r.x + fx*r.width)*dpr, (r.y + fy*r.height)*dpr)。
  - 离线：--rect "x,y,w,h,dpr" 直接给截图（物理像素）坐标系下的矩形，
    传 dpr=1 即可，例如离线样本截图用 "153,136.5,1455,826.5,1"。

CLI:
  python zmws.py rect
  python zmws.py shot <输出路径>
  python zmws.py click <fx> <fy>
  python zmws.py badges <截图路径> [--rect "x,y,w,h,dpr"]
  python zmws.py state  <截图路径> [--rect "x,y,w,h,dpr"]
  python zmws.py scan   <输出路径>            # shot + state 一步完成（提速）
  python zmws.py clickscan <fx> <fy> <等待秒> <输出路径>  # 点击+等待+截图+状态 一步完成
  python zmws.py mode                         # 查看当前策略配置
  python zmws.py mode <high|low>              # 设置仙位选卡策略
  python zmws.py xianwei <剩余次数> <工作目录>  # 仙位全自动循环（前提：对手弹窗已开、碾压已勾选）
  python zmws.py douchong <剩余次数> <工作目录>  # 斗宠全自动循环（前提：斗宠对手弹窗已开、快速挑战已勾选）

策略配置（skill 根目录 config.json）：
  xianwei_strategy = "low_power"（默认）：挑战有碾压章的最右卡（当前行为）
                     "high_power"：只挑战最右侧第 4 卡（奖励最高），
                                   无碾压章就走刷新分支，绝不打左侧低奖励卡
"""
import argparse
import json
import os
import sys
import time

# 部分机器上 OpenBLAS 默认线程数会导致内存分配失败，必须在 numpy 导入前限制
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import requests
from PIL import Image

DAEMON = "http://127.0.0.1:10086/command"
SESSION = "zaomeng-daily"

# 策略配置文件（skill 根目录 config.json）
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
DEFAULT_CONFIG = {"xianwei_strategy": "low_power", "xianwei_max_refresh": 15,
                  "douchong_max_refresh": 3}


def load_config():
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        merged = dict(DEFAULT_CONFIG)
        merged.update({k: v for k, v in cfg.items() if k in DEFAULT_CONFIG})
        return merged
    except (OSError, ValueError):
        return dict(DEFAULT_CONFIG)


def save_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 元素比例位置（标定见 SKILL.md / references/xianwei.md）
# ---------------------------------------------------------------------------
CARD_FX = [0.202, 0.415, 0.628, 0.841]  # 4 张对手卡 fx（从左到右）
FY_BADGE = 0.493          # 卡片「碾压」红章中心
FY_CHALLENGE = 0.718      # 卡片底部橙色「挑战」按钮
FX_REFRESH, FY_REFRESH = 0.927, 0.924      # 对手弹窗「刷新」
FX_OK_VICTORY, FY_OK_VICTORY = 0.502, 0.667  # 胜利结算「确定」
FX_OK_RANKUP, FY_OK_RANKUP = 0.488, 0.824    # 段位提升「确定」
FX_DROP, FY_DROP = 0.605, 0.921            # 惊喜掉落「继续游戏」（绿色）
FX_OK_CONFIRM, FY_OK_CONFIRM = 0.439, 0.554  # 刷新消耗确认弹窗「确定」（橙色，左）
FX_CANCEL_CONFIRM, FY_CANCEL_CONFIRM = 0.600, 0.554  # 同弹窗「取消」（绿色，右，永不点）

# ---------------------------------------------------------------------------
# 颜色标定常量（2026-09-14 采样，见下）
#
# 样本截图：工作区 shot-tiaozhan.jpg / shot-fight1.jpg / shot-after1.jpg
# （1781x1323 物理像素；iframe 物理矩形 x=153,y=136.5,w=1455,h=826.5）
#
# 1) 碾压红章红色（shot-tiaozhan.jpg，4 卡 fy=0.493 区域，59x41 采样框）：
#    红色像素占比 56.6%~58.9%，median RGB ≈ (160, 17, 10)，
#    范围 R[121,255] G[0,118] B[0,98]
#    → 判据 R>=120 且 R-G>=70 且 R-B>=70；区域占比阈值 0.30
# 2) 橙色按钮（挑战/刷新/确定，shot-tiaozhan/fight1/after1）：
#    橙色像素 median RGB ≈ (239, 143, 69)，范围 R[181,255] G[91,168] B[30,99]
#    → 判据 R>=180 且 80<=G<=190 且 B<=110 且 R-B>=120；区域占比阈值 0.30
# 3) 弹窗压暗背景（shot-fight1.jpg vs shot-tiaozhan.jpg 挑战按钮位置）：
#    亮度衰减到 0.26~0.31（亮按钮区域平均亮度 107~116 → 压暗后 28~33；
#    fight1 弹出的确定按钮区域平均亮度 115.8）
#    → 「亮按钮」判据：区域平均亮度 >= 55
# 4) 绿色「继续游戏」按钮：无实战样本，用通用绿色判据
#    G>=120 且 G-R>=30 且 G-B>=40；区域占比阈值 0.20（待实战校准）
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 斗宠：对手卡「战力」数字红字检测（2026-09-15 标定，见 references/douchong.md）
#
# 游戏自带信号：对手战力 > 我方时，该卡战力数字显示为红色，否则为黄色。
# 样本：dc4/dc9（全黄）、dc12/dc14（卡4 红）、dc16（卡3 红），
# 采样框 fx=卡fx-0.006±0.040、fy=0.388±0.010：
#   黄字卡 red_frac 恒 0.000；红字卡 red_frac 0.086~0.231 → 阈值 0.05
# ---------------------------------------------------------------------------
FY_DC_POWER = 0.388          # 「战力：」数字行中心
DC_FX_OFF, DC_HW, DC_HH = -0.006, 0.040, 0.010
DC_RED_MIN, DC_RED_DG, DC_RED_DB = 150, 70, 70
DC_RED_TH = 0.05

RED_MIN, RED_DG, RED_DB = 120, 70, 70
RED_FRAC_TH = 0.30
ORANGE_MIN_R, ORANGE_MIN_G, ORANGE_MAX_G, ORANGE_MAX_B, ORANGE_MIN_DB = 180, 80, 190, 110, 120
ORANGE_FRAC_TH = 0.30
GREEN_MIN, GREEN_DR, GREEN_DB = 120, 30, 40
GREEN_FRAC_TH = 0.20
BRIGHT_TH = 55.0  # 「亮按钮」区域平均亮度阈值（压暗背景约 28~33，亮按钮 107+）


def die(msg, code=2):
    print(f"zmws 错误: {msg}", file=sys.stderr)
    sys.exit(code)


def call_daemon(action, args):
    """POST 到 WebBridge daemon，返回 data 字段。"""
    payload = {"action": action, "args": args, "session": SESSION}
    try:
        resp = requests.post(DAEMON, json=payload, timeout=30)
    except requests.RequestException as e:
        die(f"无法连接 WebBridge daemon ({DAEMON}): {e}")
    if resp.status_code != 200:
        die(f"daemon HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        body = resp.json()
    except ValueError:
        die(f"daemon 返回非 JSON: {resp.text[:200]}")
    if not body.get("ok"):
        die(f"daemon 动作 {action} 失败: {json.dumps(body, ensure_ascii=False)[:300]}")
    return body.get("data")


EVAL_RECT_JS = (
    "(function(){var f=document.querySelector('iframe[src*=\"h.api.4399.com\"]');"
    "if(!f)return 'NO_IFRAME';var r=f.getBoundingClientRect();"
    "return JSON.stringify({dpr:window.devicePixelRatio,"
    "x:r.x,y:r.y,width:r.width,height:r.height});})()"
)


def get_rect_online():
    """在线取 iframe CSS 矩形 + dpr，返回 (x, y, w, h, dpr)。"""
    data = call_daemon("evaluate", {"code": EVAL_RECT_JS})
    value = (data or {}).get("value")
    if value in (None, "NO_IFRAME"):
        die("页面上找不到游戏 iframe（iframe[src*=h.api.4399.com]），请确认游戏页面已打开")
    try:
        obj = json.loads(value)
        return float(obj["x"]), float(obj["y"]), float(obj["width"]), float(obj["height"]), float(obj["dpr"])
    except (ValueError, KeyError, TypeError):
        die(f"evaluate 返回无法解析: {value!r}")


def resolve_rect(rect_arg, online=True):
    """返回 (x, y, w, h, dpr)。--rect 给出时直接使用（dpr 由调用方给定）。"""
    if rect_arg:
        try:
            parts = [float(p) for p in rect_arg.split(",")]
        except ValueError:
            die(f"--rect 格式错误: {rect_arg!r}，应为 \"x,y,w,h,dpr\"")
        if len(parts) != 5:
            die(f"--rect 需要 5 个数值 x,y,w,h,dpr，得到 {len(parts)} 个")
        return tuple(parts)
    if not online:
        die("离线分析必须提供 --rect \"x,y,w,h,dpr\"（截图物理像素矩形，dpr=1）")
    return get_rect_online()


def load_image(path):
    if not os.path.isfile(path):
        die(f"截图文件不存在: {path}")
    try:
        return np.asarray(Image.open(path).convert("RGB")).astype(np.int32)
    except Exception as e:
        die(f"无法读取截图 {path}: {e}")


def region(img, rect, fx, fy, half_w, half_h):
    """取 (fx,fy) 周围比例半宽 half_w、半高 half_h 的像素区域 (n,3)。

    rect = (x, y, w, h, dpr)：x,y,w,h 为 CSS 像素（或 --rect 给定的坐标系），
    截图像素 = (x + fx*w) * dpr。离线传 dpr=1 即直接用物理像素矩形。
    """
    x, y, w, h, dpr = rect
    cx = int(round((x + fx * w) * dpr))
    cy = int(round((y + fy * h) * dpr))
    hw = max(2, int(half_w * w * dpr))
    hh = max(2, int(half_h * h * dpr))
    ih, iw = img.shape[:2]
    x0, x1 = max(0, cx - hw), min(iw, cx + hw + 1)
    y0, y1 = max(0, cy - hh), min(ih, cy + hh + 1)
    if x1 <= x0 or y1 <= y0:
        die(f"采样区域超出截图范围: 中心=({cx},{cy}) 截图={iw}x{ih}")
    return img[y0:y1, x0:x1].reshape(-1, 3)


def is_red(reg):
    r, g, b = reg[:, 0], reg[:, 1], reg[:, 2]
    return (r >= RED_MIN) & (r - g >= RED_DG) & (r - b >= RED_DB)


def is_orange(reg):
    r, g, b = reg[:, 0], reg[:, 1], reg[:, 2]
    return (r >= ORANGE_MIN_R) & (g >= ORANGE_MIN_G) & (g <= ORANGE_MAX_G) & \
           (b <= ORANGE_MAX_B) & (r - b >= ORANGE_MIN_DB)


def is_green(reg):
    r, g, b = reg[:, 0], reg[:, 1], reg[:, 2]
    return (g >= GREEN_MIN) & (g - r >= GREEN_DR) & (g - b >= GREEN_DB)


def frac(reg, mask):
    return float(mask.mean()) if len(reg) else 0.0


def bright(reg):
    return float(reg.mean())


def lit_orange(img, rect, fx, fy, half_w, half_h):
    """返回 (是否亮橙色按钮, 橙色占比, 平均亮度)。"""
    reg = region(img, rect, fx, fy, half_w, half_h)
    f = frac(reg, is_orange(reg))
    b = bright(reg)
    return (f >= ORANGE_FRAC_TH and b >= BRIGHT_TH), f, b


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------

def cmd_rect(_args):
    x, y, w, h, dpr = get_rect_online()
    print(json.dumps({"dpr": dpr, "x": x, "y": y, "w": w, "h": h}, ensure_ascii=False))


def cmd_shot(args):
    print(do_shot(args.output))


def cmd_click(args):
    if args.rect:
        die("click 不允许使用 --rect（离线坐标点击会点到错误位置），请在线执行")
    css_x, css_y = do_click(args.fx, args.fy)
    print(json.dumps({"css_x": round(css_x, 1), "css_y": round(css_y, 1)}))


def detect_badges(img, rect):
    """返回 (badges 列表, rightmost, 各卡红色占比)。"""
    badges, fracs = [], []
    for fx in CARD_FX:
        # 红章区域：宽约 0.04*w、高约 0.05*h → 半宽 0.02、半高 0.025
        reg = region(img, rect, fx, FY_BADGE, 0.02, 0.025)
        f = frac(reg, is_red(reg))
        fracs.append(round(f, 3))
        badges.append(f >= RED_FRAC_TH)
    rightmost = 0
    for i, b in enumerate(badges):
        if b:
            rightmost = i + 1
    return badges, rightmost, fracs


def cmd_badges(args):
    rect = resolve_rect(args.rect, online=True)
    img = load_image(args.image)
    badges, rightmost, fracs = detect_badges(img, rect)
    print(json.dumps({"badges": badges, "rightmost": rightmost,
                      "red_frac": fracs}, ensure_ascii=False))


def is_dc_red(reg):
    r, g, b = reg[:, 0], reg[:, 1], reg[:, 2]
    return (r >= DC_RED_MIN) & (r - g >= DC_RED_DG) & (r - b >= DC_RED_DB)


def detect_douchong(img, rect):
    """斗宠对手卡：检测各卡战力数字是否红字（对手战力>我方）。

    返回 (dc_red 列表, dc_target, 各卡红色占比)。
    dc_target = 最右非红字卡序号（1~4），全红为 0（应走刷新分支）。
    """
    dc_red, fracs = [], []
    for fx in CARD_FX:
        reg = region(img, rect, fx + DC_FX_OFF, FY_DC_POWER, DC_HW, DC_HH)
        f = frac(reg, is_dc_red(reg))
        fracs.append(round(f, 3))
        dc_red.append(f >= DC_RED_TH)
    target = 0
    for i, red in enumerate(dc_red):
        if not red:
            target = i + 1
    return dc_red, target, fracs


def analyze(img, rect):
    """对截图做状态机判定，返回 {"state":..., "detail":...}。"""
    vic_ok, vic_f, vic_b = lit_orange(img, rect, FX_OK_VICTORY, FY_OK_VICTORY, 0.03, 0.02)
    rnk_ok, rnk_f, rnk_b = lit_orange(img, rect, FX_OK_RANKUP, FY_OK_RANKUP, 0.03, 0.02)
    drop_reg = region(img, rect, FX_DROP, FY_DROP, 0.03, 0.02)
    drop_f = frac(drop_reg, is_green(drop_reg))
    drop_b = bright(drop_reg)
    drop_ok = drop_f >= GREEN_FRAC_TH and drop_b >= BRIGHT_TH

    ch = [lit_orange(img, rect, fx, FY_CHALLENGE, 0.03, 0.015) for fx in CARD_FX]
    ch_ok = [c[0] for c in ch]

    detail = {
        "victory_btn": {"orange_frac": round(vic_f, 3), "brightness": round(vic_b, 1)},
        "rankup_btn": {"orange_frac": round(rnk_f, 3), "brightness": round(rnk_b, 1)},
        "drop_btn": {"green_frac": round(drop_f, 3), "brightness": round(drop_b, 1)},
        "challenge_btns": [{"orange_frac": round(c[1], 3), "brightness": round(c[2], 1)} for c in ch],
    }
    if vic_ok:
        state = "victory"
    elif rnk_ok:
        state = "rankup"
    elif drop_ok:
        state = "drop"
    elif all(ch_ok):
        state = "opponent_list"
        badges, rightmost, _ = detect_badges(img, rect)
        detail["badges"] = badges
        detail["rightmost"] = rightmost
        # 高战力策略：只认最右侧第 4 卡；可直接取 target 字段作为要点的卡序号
        cfg = load_config()
        if cfg.get("xianwei_strategy") == "high_power":
            detail["strategy"] = "high_power"
            detail["target"] = 4 if badges[3] else 0
        else:
            detail["strategy"] = "low_power"
            detail["target"] = rightmost
        # 斗宠：红字战力检测（对手战力>我方的卡）。dc_target=最右非红卡，全红为 0
        dc_red, dc_target, dc_fracs = detect_douchong(img, rect)
        detail["dc_red"] = dc_red
        detail["dc_red_frac"] = dc_fracs
        detail["dc_target"] = dc_target
    else:
        # 刷新消耗确认弹窗（「本次刷新将消耗xxx，是否继续」确定/取消）
        # 放在 opponent_list 之后：弹窗会压暗挑战按钮使其不满足 all(ch_ok)
        cfm_ok, cfm_f, cfm_b = lit_orange(img, rect, FX_OK_CONFIRM, FY_OK_CONFIRM, 0.03, 0.02)
        ccl_reg = region(img, rect, FX_CANCEL_CONFIRM, FY_CANCEL_CONFIRM, 0.03, 0.02)
        ccl_f = frac(ccl_reg, is_green(ccl_reg))
        ccl_b = bright(ccl_reg)
        detail["confirm_btns"] = {"ok_orange_frac": round(cfm_f, 3), "ok_brightness": round(cfm_b, 1),
                                  "cancel_green_frac": round(ccl_f, 3), "cancel_brightness": round(ccl_b, 1)}
        if cfm_ok and ccl_f >= GREEN_FRAC_TH and ccl_b >= BRIGHT_TH:
            state = "confirm"
        else:
            state = "unknown"
    return {"state": state, "detail": detail}


def cmd_state(args):
    rect = resolve_rect(args.rect, online=True)
    img = load_image(args.image)
    print(json.dumps(analyze(img, rect), ensure_ascii=False))


def do_shot(out):
    """截图到 out，返回实际文件路径。"""
    out = os.path.abspath(out)
    data = call_daemon("screenshot", {"format": "jpeg", "quality": 70, "path": out})
    path = (data or {}).get("path", out)
    if not os.path.isfile(path):
        die(f"截图后文件不存在: {path}")
    return path


def do_click(fx, fy):
    """按比例坐标点击，返回 (css_x, css_y)。"""
    x, y, w, h, dpr = get_rect_online()
    css_x = x + fx * w
    css_y = y + fy * h
    for ev in ("mousePressed", "mouseReleased"):
        call_daemon("cdp", {"method": "Input.dispatchMouseEvent", "params": {
            "type": ev, "x": round(css_x, 1), "y": round(css_y, 1),
            "button": "left", "clickCount": 1}})
        time.sleep(0.05)
    return css_x, css_y


def cmd_scan(args):
    """截图 + 状态判定一步完成（省一次进程启动和一次调用往返）。"""
    path = do_shot(args.output)
    rect = get_rect_online()
    result = analyze(load_image(path), rect)
    result["shot"] = path
    print(json.dumps(result, ensure_ascii=False))


def cmd_clickscan(args):
    """点击 → 等待 → 截图 → 状态判定，一步完成。"""
    css_x, css_y = do_click(args.fx, args.fy)
    time.sleep(max(0.0, args.wait))
    path = do_shot(args.output)
    rect = get_rect_online()
    result = analyze(load_image(path), rect)
    result["shot"] = path
    result["clicked"] = {"css_x": round(css_x, 1), "css_y": round(css_y, 1)}
    print(json.dumps(result, ensure_ascii=False))


def cmd_mode(args):
    cfg = load_config()
    if args.value is None:
        print(json.dumps(cfg, ensure_ascii=False))
        return
    mapping = {"high": "high_power", "low": "low_power"}
    cfg["xianwei_strategy"] = mapping[args.value]
    save_config(cfg)
    print(json.dumps(cfg, ensure_ascii=False))


def cmd_config(args):
    """查看/修改 config.json 任意配置项（如 xianwei_max_refresh）。"""
    cfg = load_config()
    if args.key is None:
        print(json.dumps(cfg, ensure_ascii=False))
        return
    if args.key not in DEFAULT_CONFIG:
        die(f"未知配置项: {args.key!r}，可选: {', '.join(DEFAULT_CONFIG)}")
    default = DEFAULT_CONFIG[args.key]
    try:
        cfg[args.key] = int(args.value) if isinstance(default, int) else args.value
    except ValueError:
        die(f"配置项 {args.key} 需要整数，得到 {args.value!r}")
    save_config(cfg)
    print(json.dumps(cfg, ensure_ascii=False))


def cmd_xianwei(args):
    """仙位挑战全自动循环：整个战斗过程零模型往返，只为提速。

    前提（由调用方完成）：对手弹窗已打开、「开启碾压模式」已勾选、
    已从底部状态栏读出剩余奖励次数并作为参数传入。
    循环：scan → opponent_list 按策略取 target 挑战/刷新 → victory 点确定
    → rankup/drop 自动处理，直到胜利场数达标或触发停止条件。
    停止条件：胜场达标(ok) / 刷新 15 次用尽 / unknown 画面（存图交回人工）。
    最后一行输出 "SUMMARY {json}"。
    """
    workdir = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)
    need = args.remaining
    t0 = time.time()
    shot_n = [0]

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')} +{time.time()-t0:5.1f}s] {msg}", flush=True)

    def snap(tag):
        shot_n[0] += 1
        path = os.path.join(workdir, f"xw-{tag}-{shot_n[0]:02d}.jpg")
        do_shot(path)
        return path, analyze(load_image(path), get_rect_online())

    def finish(ok, reason, wins, refreshes, events, last_shot):
        summary = {"ok": ok, "reason": reason, "wins": wins, "refreshes": refreshes,
                   "events": events, "duration_s": round(time.time() - t0, 1),
                   "last_shot": last_shot}
        print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)

    if need <= 0:
        log("剩余次数为 0，无需挑战")
        finish(True, "already_done", 0, 0, [], None)
        return

    cfg = load_config()
    strategy = cfg.get("xianwei_strategy", "low_power")
    log(f"仙位自动循环开始：目标 {need} 场，策略 {strategy}")
    wins, refreshes, steps = 0, 0, 0
    events = []
    unknown_retries = 0
    last_action = None  # 防呆：confirm 弹窗只在上一步是刷新时才自动点确定
    MAX_STEPS = 80
    MAX_REFRESH = int(cfg.get("xianwei_max_refresh", 15))  # config.json 可调

    while wins < need and steps < MAX_STEPS:
        steps += 1
        path, res = snap("loop")
        st = res["state"]
        if st == "opponent_list":
            unknown_retries = 0
            target = res["detail"].get("target", 0)
            if target >= 1:
                log(f"挑战第 {target} 卡（胜 {wins}/{need}）")
                do_click(CARD_FX[target - 1], FY_CHALLENGE)
                last_action = "challenge"
                time.sleep(4)
            else:
                if refreshes >= MAX_REFRESH:
                    log(f"刷新 {MAX_REFRESH} 次用尽仍无目标，停止")
                    finish(False, "refresh_exhausted", wins, refreshes, events, path)
                    return
                refreshes += 1
                log(f"无可挑战目标，刷新（第 {refreshes}/{MAX_REFRESH} 次）")
                do_click(FX_REFRESH, FY_REFRESH)
                last_action = "refresh"
                time.sleep(2)
        elif st == "confirm":
            unknown_retries = 0
            if last_action == "refresh":
                log("刷新消耗确认弹窗，点确定（绝不点取消旁的任何其他按钮）")
                events.append("refresh_confirm")
                do_click(FX_OK_CONFIRM, FY_OK_CONFIRM)
                last_action = None
                time.sleep(2)
            else:
                log(f"出现来源不明的确认弹窗，停止并交回人工（截图 {path}）")
                finish(False, "unexpected_confirm", wins, refreshes, events, path)
                return
        elif st == "victory":
            unknown_retries = 0
            wins += 1
            log(f"第 {wins}/{need} 场胜利，点确定")
            do_click(FX_OK_VICTORY, FY_OK_VICTORY)
            time.sleep(3)
        elif st == "rankup":
            unknown_retries = 0
            log("段位提升弹窗，点确定")
            events.append("rankup")
            do_click(FX_OK_RANKUP, FY_OK_RANKUP)
            time.sleep(3)
        elif st == "drop":
            unknown_retries = 0
            log(f"惊喜掉落弹窗，点继续游戏（截图 {path}，待人工校准判据）")
            events.append("drop")
            do_click(FX_DROP, FY_DROP)
            time.sleep(3)
        else:  # unknown
            unknown_retries += 1
            if unknown_retries <= 2:
                log(f"画面无法识别，3 秒后重试（第 {unknown_retries}/2 次，截图 {path}）")
                time.sleep(3)
            else:
                log(f"画面连续无法识别，停止并交回人工（截图 {path}）")
                finish(False, "unknown_state", wins, refreshes, events, path)
                return

    if wins >= need:
        path, _res = snap("final")
        log(f"目标完成：{wins} 场胜利，刷新 {refreshes} 次")
        finish(True, "done", wins, refreshes, events, path)
    else:
        log(f"步数达上限 {MAX_STEPS} 仍未完成，停止")
        finish(False, "step_limit", wins, refreshes, events, path)


def cmd_douchong(args):
    """斗宠挑战全自动循环：整个战斗过程零模型往返。

    前提（由调用方完成）：斗宠对手弹窗已打开、「快速挑战」已勾选、
    已从底部状态栏读出剩余奖励次数并作为参数传入。
    决策机 v1（红字策略）：挑战最右侧战力非红字（对手战力<=我方）的卡；
    全红 → 刷新（上限 douchong_max_refresh，默认 3，与游戏内每日上限一致）；
    刷新用尽仍全红 → 停止并交回人工（no_beatable）。
    注意：失败结算画面未标定，脚本会判 unknown 并停机交回人工——这是刻意的安全行为。
    最后一行输出 "SUMMARY {json}"。
    """
    workdir = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)
    need = args.remaining
    t0 = time.time()
    shot_n = [0]

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')} +{time.time()-t0:5.1f}s] {msg}", flush=True)

    def snap(tag):
        shot_n[0] += 1
        path = os.path.join(workdir, f"dc-{tag}-{shot_n[0]:02d}.jpg")
        do_shot(path)
        return path, analyze(load_image(path), get_rect_online())

    def finish(ok, reason, wins, refreshes, events, last_shot):
        summary = {"ok": ok, "reason": reason, "wins": wins, "refreshes": refreshes,
                   "events": events, "duration_s": round(time.time() - t0, 1),
                   "last_shot": last_shot}
        print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)

    if need <= 0:
        log("剩余次数为 0，无需挑战")
        finish(True, "already_done", 0, 0, [], None)
        return

    cfg = load_config()
    log(f"斗宠自动循环开始：目标 {need} 场，策略 red_power（打最右非红字卡）")
    wins, refreshes, steps = 0, 0, 0
    events = []
    unknown_retries = 0
    last_action = None  # 防呆：confirm 弹窗只在上一步是刷新时才自动点确定
    MAX_STEPS = 60
    MAX_REFRESH = int(cfg.get("douchong_max_refresh", 3))

    while wins < need and steps < MAX_STEPS:
        steps += 1
        path, res = snap("loop")
        st = res["state"]
        if st == "opponent_list":
            unknown_retries = 0
            target = res["detail"].get("dc_target", 0)
            if target >= 1:
                log(f"挑战第 {target} 卡（胜 {wins}/{need}，红字卡={res['detail'].get('dc_red')}）")
                do_click(CARD_FX[target - 1], FY_CHALLENGE)
                last_action = "challenge"
                time.sleep(5)
            else:
                if refreshes >= MAX_REFRESH:
                    log(f"4 卡全红且刷新 {MAX_REFRESH} 次用尽，停止（剩余 {need - wins} 场未打）")
                    finish(False, "no_beatable", wins, refreshes, events, path)
                    return
                refreshes += 1
                log(f"4 卡全红，刷新（第 {refreshes}/{MAX_REFRESH} 次，消耗 1000 斗宠币）")
                do_click(FX_REFRESH, FY_REFRESH)
                last_action = "refresh"
                time.sleep(2)
        elif st == "confirm":
            unknown_retries = 0
            if last_action == "refresh":
                log("刷新消耗确认弹窗，点确定")
                events.append("refresh_confirm")
                do_click(FX_OK_CONFIRM, FY_OK_CONFIRM)
                last_action = None
                time.sleep(2)
            else:
                log(f"出现来源不明的确认弹窗，停止并交回人工（截图 {path}）")
                finish(False, "unexpected_confirm", wins, refreshes, events, path)
                return
        elif st == "victory":
            unknown_retries = 0
            wins += 1
            log(f"第 {wins}/{need} 场胜利，点确定")
            do_click(FX_OK_VICTORY, FY_OK_VICTORY)
            time.sleep(3)
        elif st == "rankup":
            unknown_retries = 0
            log("段位提升弹窗，点确定")
            events.append("rankup")
            do_click(FX_OK_RANKUP, FY_OK_RANKUP)
            time.sleep(3)
        elif st == "drop":
            unknown_retries = 0
            log(f"惊喜掉落弹窗，点继续游戏（截图 {path}）")
            events.append("drop")
            do_click(FX_DROP, FY_DROP)
            time.sleep(3)
        else:  # unknown（含未标定的失败结算画面——刻意停机交回人工）
            unknown_retries += 1
            if unknown_retries <= 2:
                log(f"画面无法识别，3 秒后重试（第 {unknown_retries}/2 次，截图 {path}）")
                time.sleep(3)
            else:
                log(f"画面连续无法识别（可能是失败结算等未标定画面），停止并交回人工（截图 {path}）")
                finish(False, "unknown_state", wins, refreshes, events, path)
                return

    if wins >= need:
        path, _res = snap("final")
        log(f"目标完成：{wins} 场胜利，刷新 {refreshes} 次")
        finish(True, "done", wins, refreshes, events, path)
    else:
        log(f"步数达上限 {MAX_STEPS} 仍未完成，停止")
        finish(False, "step_limit", wins, refreshes, events, path)


def main():
    ap = argparse.ArgumentParser(prog="zmws.py", description="造梦无双日常自动化辅助脚本")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("rect", help="打印 iframe 矩形 JSON").set_defaults(fn=cmd_rect)

    p = sub.add_parser("shot", help="截图到指定路径")
    p.add_argument("output")
    p.set_defaults(fn=cmd_shot)

    p = sub.add_parser("click", help="按比例坐标点击（在线）")
    p.add_argument("fx", type=float)
    p.add_argument("fy", type=float)
    p.add_argument("--rect", default=None, help=argparse.SUPPRESS)
    p.set_defaults(fn=cmd_click)

    for name, fn in (("badges", cmd_badges), ("state", cmd_state)):
        p = sub.add_parser(name, help=f"分析截图（{name}）")
        p.add_argument("image")
        p.add_argument("--rect", default=None,
                       help='离线分析：截图坐标系下的 iframe 矩形 "x,y,w,h,dpr"（物理像素则 dpr=1）')
        p.set_defaults(fn=fn)

    p = sub.add_parser("scan", help="截图+状态判定一步完成（在线）")
    p.add_argument("output")
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("clickscan", help="点击+等待+截图+状态判定一步完成（在线）")
    p.add_argument("fx", type=float)
    p.add_argument("fy", type=float)
    p.add_argument("wait", type=float, help="点击后等待秒数")
    p.add_argument("output")
    p.set_defaults(fn=cmd_clickscan)

    p = sub.add_parser("mode", help="查看/设置仙位选卡策略")
    p.add_argument("value", nargs="?", choices=["high", "low"],
                   help="high=只打最右卡（高战力）；low=打最右可碾压卡（低战力）")
    p.set_defaults(fn=cmd_mode)

    p = sub.add_parser("config", help="查看/修改配置项（config.json）")
    p.add_argument("key", nargs="?", help="配置项名，如 xianwei_max_refresh；省略则显示全部")
    p.add_argument("value", nargs="?", help="要设置的值")
    p.set_defaults(fn=cmd_config)

    p = sub.add_parser("xianwei", help="仙位挑战全自动循环（前提：对手弹窗已开、碾压模式已勾选）")
    p.add_argument("remaining", type=int, help="剩余奖励次数（从底部状态栏读出）")
    p.add_argument("workdir", help="截图与日志输出目录")
    p.set_defaults(fn=cmd_xianwei)

    p = sub.add_parser("douchong", help="斗宠挑战全自动循环（前提：斗宠对手弹窗已开、快速挑战已勾选）")
    p.add_argument("remaining", type=int, help="剩余奖励次数（从底部状态栏读出）")
    p.add_argument("workdir", help="截图与日志输出目录")
    p.set_defaults(fn=cmd_douchong)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
