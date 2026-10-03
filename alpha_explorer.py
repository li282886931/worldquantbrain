"""WorldQuant BRAIN alpha expression explorer.

Reads BRAIN credentials (WQB_USERNAME / WQB_PASSWORD) from the project .env
file via ``config.load_wqb_credentials`` and iterates over data fields in a
dataset, generating and testing alpha expressions with multiple settings
combinations.

Run directly::

    python alpha_explorer.py
"""

import pickle
import random
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

from config import load_wqb_credentials

# ------------- 全局配置 -------------
DATASET_ID = "pv13"
DATA_TYPE = "MATRIX"

MIN_SHARPE = 1.25
MIN_FITNESS = 1.0
PROGRESS_FILE = "alpha_progress_fine.pkl"

MAX_SETTINGS_PER_EXPR = 5
SIMULATIONS_PER_BATCH = 3

QUICK_SETTINGS = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 0,
    "neutralization": "MARKET",
    "truncation": 0.08,
    "pasteurization": "ON",
    "unitHandling": "VERIFY",
    "nanHandling": "ON",
    "language": "FASTEXPR",
    "visualization": False,
}

PRE_FILTER_PASSED_FLAG = -2
MAX_POLL_ATTEMPTS = 200  # 模拟进度轮询上限，防止死循环


# ------------- Session 与请求 -------------
def make_session(username: str, password: str) -> requests.Session:
    sess = requests.Session()
    sess.auth = HTTPBasicAuth(username, password)
    return sess


def refresh_session(sess, username, password):
    """自动重新登录，刷新 sess 凭证。"""
    sess.auth = HTTPBasicAuth(username, password)
    print("🔄 正在重新登录 WorldQuant BRAIN...")
    resp = sess.post("https://api.worldquantbrain.com/authentication")
    if resp.status_code != 201:
        print(f"❌ 重登失败，状态码：{resp.status_code}")
        raise SystemExit("重登失败，请检查账号或网络")
    print("✅ 凭证刷新成功")


def safe_request(sess, method, url, username, password, **kwargs):
    """带自动重连的请求函数。"""
    resp = sess.request(method, url, **kwargs)
    if resp.status_code == 401:
        refresh_session(sess, username, password)
        resp = sess.request(method, url, **kwargs)
    return resp


# ------------- 核心函数 -------------
def get_datafields(sess, username, password, dataset_id="pv13", data_type="MATRIX"):
    offset = 0
    all_fields = []
    while True:
        url = (
            "https://api.worldquantbrain.com/data-fields?"
            "&instrumentType=EQUITY&region=USA&delay=1&universe=TOP3000"
            f"&dataset.id={dataset_id}&limit=50&offset={offset}&type={data_type}"
        )
        resp = safe_request(sess, "GET", url, username, password)
        data = resp.json()
        if "results" not in data:
            break
        batch = data["results"]
        print(f"📦 获取到 {len(batch)} 个字段，偏移量 {offset}")
        all_fields.append(batch)
        if len(batch) < 50:
            break
        offset += 50
        time.sleep(5)
    flat_list = [item for sublist in all_fields for item in sublist]
    return pd.DataFrame(flat_list)


def submit_alpha(sess, username, password, expression, settings=None, max_retries=5):
    if settings is None:
        settings = {
            "instrumentType": "EQUITY",
            "region": "USA",
            "universe": "TOP3000",
            "delay": 1,
            "decay": 0,
            "neutralization": "MARKET",
            "truncation": 0.08,
            "pasteurization": "ON",
            "unitHandling": "VERIFY",
            "nanHandling": "ON",
            "language": "FASTEXPR",
            "visualization": False,
        }
    sim_data = {"type": "REGULAR", "settings": settings, "regular": expression}
    for attempt in range(max_retries):
        try:
            resp = safe_request(
                sess,
                "POST",
                "https://api.worldquantbrain.com/simulations",
                username,
                password,
                json=sim_data,
            )
            if "Location" in resp.headers:
                progress_url = resp.headers["Location"]
                for _ in range(MAX_POLL_ATTEMPTS):
                    time.sleep(3)
                    try:
                        p_data = safe_request(
                            sess, "GET", progress_url, username, password
                        ).json()
                    except Exception:
                        continue
                    if "alpha" in p_data:
                        return {"status": "success", "alpha_id": p_data["alpha"]}
                    if "progress" in p_data and p_data.get("status") != "ERROR":
                        continue
                    return {"status": "failed", "message": str(p_data)}
                return {"status": "failed", "message": "轮询超时，模拟未完成"}
            try:
                data = resp.json()
            except Exception:
                time.sleep(15)
                continue
            if "alpha" in data:
                return {"status": "success", "alpha_id": data["alpha"]}
            if "detail" in data and "CONCURRENT_SIMULATION_LIMIT_EXCEEDED" in data["detail"]:
                wait = 20 * (attempt + 1)
                print(f"⏳ 并发限制，等待 {wait} 秒后重试...")
                time.sleep(wait)
                continue
            if "detail" in data:
                return {"status": "failed", "message": data["detail"]}
            if "id" in data:
                task_id = data["id"]
                progress_url = f"https://api.worldquantbrain.com/simulations/{task_id}"
                for _ in range(MAX_POLL_ATTEMPTS):
                    time.sleep(3)
                    try:
                        p_data = safe_request(
                            sess, "GET", progress_url, username, password
                        ).json()
                    except Exception:
                        continue
                    if "alpha" in p_data:
                        return {"status": "success", "alpha_id": p_data["alpha"]}
                    if "progress" in p_data and p_data.get("status") != "ERROR":
                        continue
                    return {"status": "failed", "message": str(p_data)}
                return {"status": "failed", "message": "轮询超时，模拟未完成"}
            return {"status": "failed", "message": f"未知响应: {data}"}
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️ 异常 ({e})，20秒后重试...")
                time.sleep(20)
            else:
                return {"status": "failed", "message": f"重试{max_retries}次后仍失败: {e}"}
    return {"status": "failed", "message": "超过最大重试次数"}


def get_alpha_details(sess, username, password, alpha_id, max_retries=5):
    """返回 (sharpe, fitness, turnover, long_sharpe, short_sharpe)。"""
    url = f"https://api.worldquantbrain.com/alphas/{alpha_id}"
    for _ in range(max_retries):
        try:
            data = safe_request(sess, "GET", url, username, password).json()
            is_data = data.get("is", {})
            sharpe = is_data.get("sharpe")
            fitness = is_data.get("fitness")
            turnover = is_data.get("turnover")
            long_sharpe = is_data.get("longSharpe")
            short_sharpe = is_data.get("shortSharpe")
            if sharpe is not None and fitness is not None:
                return sharpe, fitness, turnover, long_sharpe, short_sharpe
            time.sleep(5)
        except Exception:
            time.sleep(5)
    return None, None, None, None, None


# ------------- 字段分类 -------------
def classify_field(field_name):
    f = field_name.lower()
    if any(kw in f for kw in ["price", "close", "open", "high", "low", "vwap", "mid"]):
        return "price"
    if any(kw in f for kw in ["volume", "turnover", "vol", "adv", "liquidity"]):
        return "volume"
    if any(kw in f for kw in ["return", "ret", "rtn", "chg", "pct"]):
        return "return"
    if any(
        kw in f
        for kw in [
            "asset", "equity", "debt", "liability", "income", "revenue",
            "earn", "profit", "margin", "roe", "roa", "eps", "pe", "pb",
            "bv", "cf", "fcf", "div", "yield", "growth",
        ]
    ):
        return "fundamental"
    return "other"


# ------------- 表达式模板 -------------
def generate_expressions(field, field_type):
    templates = []
    if field_type in ["price", "return"]:
        templates.extend([
            f"reverse(ts_delta({field}, 5))",
            f"ts_mean({field}, 5) / ts_mean({field}, 63)",
            f"ts_std_dev({field}, 20)",
            f"ts_rank({field}, 252)",
            f"ts_av_diff({field}, 20)",
            f"signed_power(ts_av_diff({field}, 20), 1.5)",
            f"ts_arg_max({field}, 63)",
        ])
    if field_type == "volume":
        templates.extend([
            f"ts_corr({field}, close, 20)",
            f"multiply(ts_delta({field}, 10), ts_delta(close, 10))",
            f"and(ts_delta({field}, 10) > 0, ts_delta(close, 10) > 0)",
        ])
    if field_type == "fundamental":
        templates.extend([
            f"ts_rank({field}, 252)",
            f"ts_zscore({field}, 252)",
            f"ts_decay_linear({field}, 126)",
            f"divide({field}, close)",
        ])
    # 通用模板
    templates.extend([
        f"rank({field})",
        f"zscore({field})",
        f"ts_zscore({field}, 63)",
        f"ts_delta({field}, 20)",
        f"ts_sum({field}, 20)",
        f"ts_delay({field}, 1)",
    ])
    return list(set(templates))


# ------------- Settings 搜索空间 -------------
def get_settings_search_space():
    base = {
        "instrumentType": "EQUITY",
        "region": "USA",
        "universe": "TOP3000",
        "language": "FASTEXPR",
        "visualization": False,
        "unitHandling": "VERIFY",
    }
    delays = [1]
    decays = [0, 3, 7]
    neutralizations = ["MARKET", "SECTOR", "INDUSTRY"]
    truncations = [0.05, 0.08, 0.1]
    pasteurizations = ["ON", "OFF"]
    nan_handlings = ["ON", "OFF"]
    combos = []
    for delay, decay, neut, trunc, past, nan in product(
        delays, decays, neutralizations, truncations, pasteurizations, nan_handlings
    ):
        settings = base.copy()
        settings["delay"] = delay
        settings["decay"] = decay
        settings["neutralization"] = neut
        settings["truncation"] = trunc
        settings["pasteurization"] = past
        settings["nanHandling"] = nan
        combos.append(settings)
    return combos


def compare_settings_for_expression(
    sess, username, password, expr, settings_list, max_combos=10
):
    total_pool = list(settings_list)
    random.shuffle(total_pool)

    if len(total_pool) > max_combos:
        initial_batch = total_pool[:max_combos]
        reserve_pool = total_pool[max_combos:]
    else:
        initial_batch = total_pool
        reserve_pool = []

    tested_list = list(initial_batch)
    extra_triggered = False
    best_id, best_sharpe, best_fitness = None, -999, -999
    best_settings = None

    def submit_with_own_session(cfg):
        worker_sess = make_session(username, password)
        worker_sess.headers.update(sess.headers)
        worker_sess.cookies.update(sess.cookies)
        try:
            return submit_alpha(
                worker_sess, username, password, expr, settings=cfg
            )
        finally:
            worker_sess.close()

    idx = 0
    while idx < len(tested_list):
        batch = tested_list[idx : idx + SIMULATIONS_PER_BATCH]
        idx += len(batch)
        for cfg in batch:
            print(
                f"    ▶ 尝试 Settings: delay={cfg['delay']}, decay={cfg['decay']}, "
                f"neut={cfg['neutralization']}, trunc={cfg['truncation']}, "
                f"past={cfg['pasteurization']}, nan={cfg['nanHandling']}"
            )

        with ThreadPoolExecutor(max_workers=SIMULATIONS_PER_BATCH) as executor:
            results = list(executor.map(submit_with_own_session, batch))

        for cfg, res in zip(batch, results):
            if res["status"] == "success":
                alpha_id = res["alpha_id"]
                sharpe, fitness, *_ = get_alpha_details(
                    sess, username, password, alpha_id
                )
                if sharpe is not None and fitness is not None:
                    if not extra_triggered and sharpe >= 1.1 and fitness >= 0.9:
                        extra_triggered = True
                        extra_count = min(5, len(reserve_pool))
                        if extra_count > 0:
                            extra_samples = random.sample(reserve_pool, extra_count)
                            tested_list.extend(extra_samples)
                            for s in extra_samples:
                                reserve_pool.remove(s)
                            print(
                                f"    🔁 触发追加调试：Sharpe={sharpe:.2f}, "
                                f"Fitness={fitness:.2f} >= 阈值，再测 {extra_count} 组设置"
                            )

                    if sharpe > 1.25 and fitness > 1.0:
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_fitness = fitness
                            best_id = alpha_id
                            best_settings = cfg
            time.sleep(2)

    if best_id:
        return best_id, (best_sharpe, best_fitness, None, None, None), best_settings
    return None, None, None


# ------------- 进度持久化 -------------
def save_progress(field_idx, expr_idx, good_alphas, skipped_fields):
    with open(PROGRESS_FILE, "wb") as f:
        pickle.dump(
            {
                "last_field_idx": field_idx,
                "last_expr_idx": expr_idx,
                "good_alphas": good_alphas,
                "skipped_fields": skipped_fields,
            },
            f,
        )


def load_progress(total_fields):
    try:
        with open(PROGRESS_FILE, "rb") as f:
            state = pickle.load(f)
            last_field_idx = state.get("last_field_idx", -1)
            last_expr_idx = state.get("last_expr_idx", -1)
            good_alphas = state.get("good_alphas", [])
            skipped_fields = state.get("skipped_fields", [])
        if last_field_idx >= total_fields:
            print("检测到上次已完成所有字段，将从头开始（如需避免请删除进度文件）")
            return -1, -1, [], []
        print(
            f"✅ 从断点恢复：字段索引 {last_field_idx}，"
            f"字段内表达式索引 {last_expr_idx}"
        )
        print(f"   已收集 {len(good_alphas)} 个优质 Alpha")
        return last_field_idx, last_expr_idx, good_alphas, skipped_fields
    except FileNotFoundError:
        print("没有进度文件，从头开始处理所有字段\n")
        return -1, -1, [], []


def build_expressions(field):
    ftype = classify_field(field)
    bases = generate_expressions(field, ftype)
    exprs = []
    for base in bases:
        for group_op in ["group_neutralize", "group_mean", "group_rank"]:
            for group_dim in ["market", "sector", "industry"]:
                exprs.append(f"{group_op}({base}, {group_dim})")
    return exprs


# ------------- 主流程 -------------
def main():
    # 从 .env 加载账号密码
    username, password = load_wqb_credentials()

    sess = make_session(username, password)

    print("正在登录 WorldQuant BRAIN...")
    resp = sess.post("https://api.worldquantbrain.com/authentication")
    if resp.status_code != 201:
        print(f"❌ 登录失败，状态码：{resp.status_code}")
        raise SystemExit("登录失败")
    print("✅ 登录成功！\n")

    settings_space = get_settings_search_space()

    # 获取字段列表
    print(f"从数据集 '{DATASET_ID}' 获取数据字段...")
    fields_df = get_datafields(
        sess, username, password, dataset_id=DATASET_ID, data_type=DATA_TYPE
    )
    if fields_df.empty:
        print("❌ 未获取到任何字段")
        raise SystemExit("字段列表为空")

    datafields = fields_df["id"].values
    print(f"✅ 共获取 {len(datafields)} 个字段\n")

    # 断点续跑
    last_field_idx, last_expr_idx, good_alphas, skipped_fields = load_progress(
        len(datafields)
    )

    total_expr_run = 0

    # 主循环
    field_idx = last_field_idx if last_field_idx >= 0 else 0
    while field_idx < len(datafields):
        field = datafields[field_idx]

        need_continue = True
        if field_idx == last_field_idx:
            if last_expr_idx >= 0:
                # 部分表达式已完成：恢复表达式列表，跳过预筛选
                exprs = build_expressions(field)
                start_expr = last_expr_idx + 1
                print(
                    f"\n--- 字段 [{field_idx+1}/{len(datafields)}] : {field} "
                    f"（从表达式 {start_expr+1} 继续） ---"
                )
            elif last_expr_idx == PRE_FILTER_PASSED_FLAG:
                # 预筛选已通过，但表达式尚未开始
                exprs = build_expressions(field)
                start_expr = 0
                print(
                    f"\n--- 字段 [{field_idx+1}/{len(datafields)}] : {field} "
                    f"（预筛选已通过，继续表达式测试） ---"
                )
            else:
                # last_expr_idx == -1 表示该字段已完成或跳过
                field_idx += 1
                continue
        else:
            # 新字段
            print(f"\n--- 字段 [{field_idx+1}/{len(datafields)}] : {field} ---")
            test_expr = f"rank({field})"
            print(f"  预筛选测试: {test_expr} … ", end="")
            res = submit_alpha(sess, username, password, test_expr, settings=QUICK_SETTINGS)
            if res["status"] != "success":
                print("⚠️ 提交失败，跳过该字段")
                skipped_fields.append(field)
                save_progress(field_idx + 1, -1, good_alphas, skipped_fields)
                field_idx += 1
                continue

            alpha_id = res["alpha_id"]
            sharpe, fitness, *_ = get_alpha_details(sess, username, password, alpha_id)
            if sharpe is None:
                print("⚠️ 无法获取指标，跳过该字段")
                skipped_fields.append(field)
                save_progress(field_idx + 1, -1, good_alphas, skipped_fields)
                field_idx += 1
                continue

            if sharpe < 0:
                print(f"❌ Sharpe={sharpe:.2f} < 0，跳过该字段全部表达式")
                skipped_fields.append(field)
                save_progress(field_idx + 1, -1, good_alphas, skipped_fields)
                field_idx += 1
                continue

            print(f"✅ Sharpe={sharpe:.2f}，字段通过预筛选，开始生成并测试表达式...")

            exprs = build_expressions(field)
            start_expr = 0
            # 预筛选通过，保存状态
            save_progress(field_idx, PRE_FILTER_PASSED_FLAG, good_alphas, skipped_fields)

        # 遍历表达式
        print(f"  该字段共 {len(exprs)} 个表达式，从第 {start_expr+1} 个开始")
        for expr_idx in range(start_expr, len(exprs)):
            expr = exprs[expr_idx]
            total_expr_run += 1
            print(f"    [{expr_idx+1}/{len(exprs)}] {expr}")
            best_id, best_details, best_cfg = compare_settings_for_expression(
                sess,
                username,
                password,
                expr,
                settings_space,
                max_combos=MAX_SETTINGS_PER_EXPR,
            )
            if best_id is not None:
                s, f, *_ = best_details
                print(
                    f"      ✅ 最优 Settings: delay={best_cfg['delay']}, "
                    f"decay={best_cfg['decay']}, neut={best_cfg['neutralization']}, "
                    f"trunc={best_cfg['truncation']}, Sharpe={s:.2f}, Fitness={f:.2f}"
                )
                good_alphas.append(
                    {
                        "expression": expr,
                        "alpha_id": best_id,
                        "sharpe": s,
                        "fitness": f,
                        "settings": best_cfg,
                    }
                )
                print(f"      🎉 优质 Alpha 入选！当前总数 {len(good_alphas)}")
            else:
                print("      ❌ 未找到满足高阈值条件的 Alpha")

            # 表达式完成，保存进度
            save_progress(field_idx, expr_idx, good_alphas, skipped_fields)

        # 字段完成
        save_progress(field_idx + 1, -1, good_alphas, skipped_fields)
        field_idx += 1

    # 输出结果
    print(f"\n===== 探索完成 =====")
    print(f"总共运行表达式数: {total_expr_run}")
    print(f"优质 Alpha 数: {len(good_alphas)}")
    print(f"跳过的字段数: {len(skipped_fields)}")
    if skipped_fields:
        print("跳过的字段列表:", skipped_fields)
    for a in good_alphas:
        print(
            f"ID: {a['alpha_id']}, Sharpe: {a['sharpe']:.2f}, "
            f"Fitness: {a['fitness']:.2f}, 最优Settings: {a['settings']}"
        )
    if not good_alphas:
        print("未筛选出优质 Alpha，可尝试调整阈值或更换数据集。")


if __name__ == "__main__":
    main()
