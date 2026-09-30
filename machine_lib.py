from collections import defaultdict, deque
from datetime import datetime
from itertools import product
import math
import re
from time import sleep
import time
from urllib.parse import urljoin

import pandas as pd
import requests

from config import load_wqb_credentials


API_BASE_URL = "https://api.worldquantbrain.com"
DEFAULT_TIMEOUT = 30
PAGE_SIZE = 50


# 将添加获得权限的Vector操作符添加在此处
 
basic_ops = ["reverse", "inverse", "rank", "zscore", "quantile", "normalize"]
 
ts_ops = ["ts_rank", "ts_zscore", "ts_delta",  "ts_sum", "ts_delay", 
          "ts_std_dev", "ts_mean",  "ts_arg_min", "ts_arg_max","ts_scale", "ts_quantile"]
 
ops_set = basic_ops + ts_ops 

def login():
    """Authenticate and return a reusable WorldQuant Brain session."""
    username, password = load_wqb_credentials()

    session = requests.Session()
    session.auth = (username, password)
    response = session.post(
        f"{API_BASE_URL}/authentication",
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    return session


def get_datasets(
    s,
    instrument_type: str = 'EQUITY',
    region: str = 'USA',
    delay: int = 1,
    universe: str = 'TOP3000'
):
    params = {
        "instrumentType": instrument_type,
        "region": region,
        "delay": delay,
        "universe": universe,
    }
    response = s.get(
        f"{API_BASE_URL}/data-sets",
        params=params,
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    return pd.DataFrame(response.json().get("results", []))


def get_datafields(
    s,
    instrument_type: str = 'EQUITY',
    region: str = 'USA',
    delay: int = 1,
    universe: str = 'TOP3000',
    dataset_id: str = '',
    search: str = ''
):
    params = {
        "instrumentType": instrument_type,
        "region": region,
        "delay": delay,
        "universe": universe,
        "limit": PAGE_SIZE,
    }
    if search:
        params["search"] = search
    elif dataset_id:
        params["dataset.id"] = dataset_id

    records = []
    offset = 0
    while True:
        page_params = {**params, "offset": offset}
        response = s.get(
            f"{API_BASE_URL}/data-fields",
            params=page_params,
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        records.extend(payload.get("results", []))
        count = int(payload.get("count", len(records)))
        offset += PAGE_SIZE
        if offset >= count:
            break

    return pd.DataFrame(records)


def get_vec_fields(fields):

    # 请在此处添加获得权限的Vector操作符
    vec_ops = ["vec_avg", "vec_sum"]
    vec_fields = []
 
    for field in fields:
        for vec_op in vec_ops:
            if vec_op == "vec_choose":
                vec_fields.append("%s(%s, nth=-1)"%(vec_op, field))
                vec_fields.append("%s(%s, nth=0)"%(vec_op, field))
            else:
                vec_fields.append("%s(%s)"%(vec_op, field))
 
    return(vec_fields)


def process_datafields(df):
    required_columns = {"id", "type"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            "Data fields must include id and type columns; "
            f"missing: {', '.join(sorted(missing))}"
        )
    datafields = []
    datafields += df[df['type'] == "MATRIX"]["id"].tolist()
    datafields += get_vec_fields(df[df['type'] == "VECTOR"]["id"].tolist())
    return ["winsorize(ts_backfill(%s, 120), std=4)"%field for field in datafields]


def ts_factory(op, field):
    output = []
    #days = [3, 5, 10, 20, 60, 120, 240]
    days = [5, 22, 66, 120, 240]
    
    for day in days:
    
        alpha = "%s(%s, %d)"%(op, field, day)
        output.append(alpha)
    
    return output

def first_order_factory(fields, ops_set):
    alpha_set = []
    #for field in fields:
    for field in fields:
        #reverse op does the work
        alpha_set.append(field)
        #alpha_set.append("-%s"%field)
        for op in ops_set:
 
            if op == "ts_percentage":
 
                alpha_set += ts_comp_factory(op, field, "percentage", [0.5])
 
 
            elif op == "ts_decay_exp_window":
 
                alpha_set += ts_comp_factory(op, field, "factor", [0.5])
 
 
            elif op == "ts_moment":
 
                alpha_set += ts_comp_factory(op, field, "k", [2, 3, 4])
 
            elif op == "ts_entropy":
 
                alpha_set += ts_comp_factory(op, field, "buckets", [10])
 
            elif op.startswith("ts_") or op == "inst_tvr":
 
                alpha_set += ts_factory(op, field)
 
            elif op.startswith("vector"):
 
                alpha_set += vector_factory(op, field)
 
            elif op == "signed_power":
 
                alpha = "%s(%s, 2)"%(op, field)
                alpha_set.append(alpha)
 
            else:
                alpha = "%s(%s)"%(op, field)
                alpha_set.append(alpha)
 
    return alpha_set

def load_task_pool_single(alpha_list, limit_of_single_simulations):

    '''
    Input:
        alpha_list : list of (alpha, decay) tuples
        limit_of_single_simulations : number of concurrent single simulations
    Output:
        task : [3 * (alpha, decay)] for 3 single simulations
        pool : [ alpha_num/3 * [3 * (alpha, decay)] ] 
    '''

    if limit_of_single_simulations <= 0:
        raise ValueError("limit_of_single_simulations must be positive")
    pool = [
        alpha_list[i:i + limit_of_single_simulations]
        for i in range(0, len(alpha_list), limit_of_single_simulations)
    ]
    return pool


def single_simulate(alpha_pool, neut, region, universe, start):
    s = login()
    failures = []

    for x, task in enumerate(alpha_pool):
        if x < start:
            continue
        progress_urls = []
        for alpha, decay in task:
            simulation_data = {
                'type': 'REGULAR',
                'settings': {
                    'instrumentType': 'EQUITY',
                    'region': region, 
                    'universe': universe, 
                    'delay': 1,
                    'decay': decay, 
                    'neutralization': neut,
                    'truncation': 0.08,
                    'pasteurization': 'ON',
                    'testPeriod': 'P0Y',
                    'unitHandling': 'VERIFY',
                    'nanHandling': 'ON',
                    'language': 'FASTEXPR',
                    'visualization': False,
                },
                'regular': alpha,
            }

            try:
                response = s.post(
                    f"{API_BASE_URL}/simulations",
                    json=simulation_data,
                    timeout=DEFAULT_TIMEOUT,
                )
                response.raise_for_status()
                progress_url = response.headers.get("Location")
                if not progress_url:
                    raise KeyError("Missing Location header")
                progress_urls.append((alpha, progress_url))
            except (requests.RequestException, KeyError) as exc:
                failures.append((alpha, str(exc).strip("'")))
                print(f"simulation post failed for {alpha}: {exc}")

        print("task %d post done"%(x))

        for alpha, progress in progress_urls:
            try:
                while True:
                    simulation_progress = s.get(
                        progress,
                        timeout=DEFAULT_TIMEOUT,
                    )
                    retry_after = simulation_progress.headers.get("Retry-After")
                    if not retry_after:
                        break
                    sleep(float(retry_after))

                simulation_progress.raise_for_status()
                status = simulation_progress.json().get("status", 0)
                if status not in {"COMPLETE", "WARNING"}:
                    print("Not complete : %s"%(progress))
                    failures.append((alpha, f"status={status}"))
            except (requests.RequestException, ValueError, TypeError) as exc:
                failures.append((alpha, str(exc)))
                print(f"simulation check failed for {progress}: {exc}")

        print("task %d simulate done"%(x))
    
    print("Simulate done")
    return failures


def set_alpha_properties(
    s,
    alpha_id,
    name: str = None,
    color: str = None,
    selection_desc: str = "None",
    combo_desc: str = "None",
    tags=None,
):
    """
    Function changes alpha's description parameters
    """
 
    params = {
        "color": color,
        "name": name,
        "tags": ["ace_tag"] if tags is None else tags,
        "category": None,
        "regular": {"description": None},
        "combo": {"description": combo_desc},
        "selection": {"description": selection_desc},
    }
    response = s.patch(
        f"{API_BASE_URL}/alphas/{alpha_id}",
        json=params,
        timeout=DEFAULT_TIMEOUT,
    )
    response.raise_for_status()
    return response


def get_alphas(start_date, end_date, sharpe_th, fitness_th, region, alpha_num, usage):
    s = login()
    output = []
    count = 0

    def normalize_date(value):
        formats = ("%Y-%m-%d", "%m-%d")
        for date_format in formats:
            try:
                parsed = datetime.strptime(value, date_format)
                if date_format == "%m-%d":
                    parsed = parsed.replace(year=datetime.now().year)
                return parsed.strftime("%Y-%m-%dT00:00:00-04:00")
            except ValueError:
                continue
        raise ValueError(
            f"Invalid date {value!r}; expected YYYY-MM-DD or MM-DD"
        )

    start_timestamp = normalize_date(start_date)
    end_timestamp = normalize_date(end_date)

    for i in range(0, alpha_num, 100):
        print(i)
        common_params = {
            "limit": 100,
            "offset": i,
            "status": "UNSUBMITTED\x1fIS_FAIL",
            "dateCreated>": start_timestamp,
            "dateCreated<": end_timestamp,
            "settings.region": region,
            "hidden": "false",
            "type!=": "SUPER",
        }
        positive_params = {
            **common_params,
            "is.fitness>": fitness_th,
            "is.sharpe>": sharpe_th,
            "order": "-is.sharpe",
        }
        queries = [positive_params]
        if usage != "submit":
            queries.append({
                **common_params,
                "is.fitness<": -fitness_th,
                "is.sharpe<": -sharpe_th,
                "order": "is.sharpe",
            })
        for params in queries:
            try:
                response = s.get(
                    f"{API_BASE_URL}/users/self/alphas",
                    params=params,
                    timeout=DEFAULT_TIMEOUT,
                )
                response.raise_for_status()
                alpha_list = response.json().get("results", [])
                for alpha in alpha_list:
                    alpha_id = alpha["id"]
                    dateCreated = alpha["dateCreated"]
                    sharpe = alpha["is"]["sharpe"]
                    fitness = alpha["is"]["fitness"]
                    turnover = alpha["is"]["turnover"]
                    margin = alpha["is"]["margin"]
                    longCount = alpha["is"]["longCount"]
                    shortCount = alpha["is"]["shortCount"]
                    decay = alpha["settings"]["decay"]
                    exp = alpha["regular"]["code"]
                    count += 1
                    if (longCount + shortCount) > 100:
                        if sharpe < -sharpe_th:
                            exp = "-%s"%exp
                        rec = [alpha_id, exp, sharpe, turnover, fitness, margin, dateCreated, decay]
                        print(rec)
                        if turnover > 0.7:
                            rec.append(decay*4)
                        elif turnover > 0.6:
                            rec.append(decay*3+3)
                        elif turnover > 0.5:
                            rec.append(decay*3)
                        elif turnover > 0.4:
                            rec.append(decay*2)
                        elif turnover > 0.35:
                            rec.append(decay+4)
                        elif turnover > 0.3:
                            rec.append(decay+2)
                        output.append(rec)
            except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
                print(f"{i} query failed, re-login: {exc}")
                s = login()

    print("count: %d"%count)
    return output

def prune(next_alpha_recs, prefix, keep_num):
    # prefix is the datafield prefix, fnd6, mdl175 ...
    # keep_num is the num of top sharpe same-datafield alpha
    output = []
    num_dict = defaultdict(int)
    for rec in next_alpha_recs:
        exp = rec[1]
        match = re.search(rf"\b{re.escape(prefix)}[A-Za-z0-9_]*", exp)
        field = match.group(0) if match else exp
        sharpe = rec[2]
        if sharpe < 0:
            field = "-%s"%field
        if num_dict[field] < keep_num:
            num_dict[field] += 1
            decay = rec[-1]
            exp = rec[1]
            output.append([exp,decay])
    return output

def get_group_second_order_factory(first_order, group_ops, region):
    second_order = []
    for fo in first_order:
        for group_op in group_ops:
            second_order += group_factory(group_op, fo, region)
    return second_order


def group_factory(op, field, region):
    output = []
    vectors = ["cap"] 
    
    usa_groups = [
        "pv13_h_min2_3000_sector",
        "pv13_r2_min20_3000_sector",
        "pv13_r2_min2_3000_sector",
        "pv13_h_min2_focused_pureplay_3000_sector",
    ]
    
    cap_group = "bucket(rank(cap), range='0.1, 1, 0.1')"
    asset_group = "bucket(rank(assets),range='0.1, 1, 0.1')"
    sector_cap_group = "bucket(group_rank(cap, sector),range='0.1, 1, 0.1')"
    sector_asset_group = "bucket(group_rank(assets, sector),range='0.1, 1, 0.1')"

    vol_group = "bucket(rank(ts_std_dev(returns,20)),range = '0.1, 1, 0.1')"

    liquidity_group = "bucket(rank(close*volume),range = '0.1, 1, 0.1')"

    groups = ["market","sector", "industry", "subindustry",
              cap_group, asset_group, sector_cap_group, sector_asset_group, vol_group, liquidity_group]
    
    if region.upper() == "USA":
        groups += usa_groups
        
    for group in groups:
        if op.startswith("group_vector"):
            for vector in vectors:
                alpha = "%s(%s,%s,densify(%s))"%(op, field, vector, group)
                output.append(alpha)
        elif op.startswith("group_percentage"):
            alpha = "%s(%s,densify(%s),percentage=0.5)"%(op, field, group)
            output.append(alpha)
        else:
            alpha = "%s(%s,densify(%s))"%(op, field, group)
            output.append(alpha)
        
    return output

def trade_when_factory(op,field,region):
    output = []
    open_events = ["ts_arg_max(volume, 5) == 0", "ts_corr(close, volume, 20) < 0",
                   "ts_corr(close, volume, 5) < 0", "ts_mean(volume,10)>ts_mean(volume,60)",
                   "group_rank(ts_std_dev(returns,60), sector) > 0.7", "ts_zscore(returns,60) > 2",
                   "ts_arg_min(volume, 5) > 3",
                   "ts_std_dev(returns, 5) > ts_std_dev(returns, 20)",
                   "ts_arg_max(close, 5) == 0", "ts_arg_max(close, 20) == 0",
                   "ts_corr(close, volume, 5) > 0", "ts_corr(close, volume, 5) > 0.3", "ts_corr(close, volume, 5) > 0.5",
                   "ts_corr(close, volume, 20) > 0", "ts_corr(close, volume, 20) > 0.3", "ts_corr(close, volume, 20) > 0.5",
                   "ts_regression(returns, %s, 5, lag = 0, rettype = 2) > 0"%field,
                   "ts_regression(returns, %s, 20, lag = 0, rettype = 2) > 0"%field,
                   "ts_regression(returns, ts_step(20), 20, lag = 0, rettype = 2) > 0",
                   "ts_regression(returns, ts_step(5), 5, lag = 0, rettype = 2) > 0"]

    exit_events = ["abs(returns) > 0.1", "-1"]

    usa_events = ["rank(rp_css_business) > 0.8", "ts_rank(rp_css_business, 22) > 0.8", "rank(vec_avg(mws82_sentiment)) > 0.8",
                  "ts_rank(vec_avg(mws82_sentiment),22) > 0.8", "rank(vec_avg(nws48_ssc)) > 0.8",
                  "ts_rank(vec_avg(nws48_ssc),22) > 0.8", "rank(vec_avg(mws50_ssc)) > 0.8", "ts_rank(vec_avg(mws50_ssc),22) > 0.8",
                  "ts_rank(vec_sum(scl12_alltype_buzzvec),22) > 0.9", "pcr_oi_270 < 1", "pcr_oi_270 > 1",]

    asi_events = ["rank(vec_avg(mws38_score)) > 0.8", "ts_rank(vec_avg(mws38_score),22) > 0.8"]

    eur_events = ["rank(rp_css_business) > 0.8", "ts_rank(rp_css_business, 22) > 0.8",
                  "rank(vec_avg(oth429_research_reports_fundamental_keywords_4_method_2_pos)) > 0.8",
                  "ts_rank(vec_avg(oth429_research_reports_fundamental_keywords_4_method_2_pos),22) > 0.8",
                  "rank(vec_avg(mws84_sentiment)) > 0.8", "ts_rank(vec_avg(mws84_sentiment),22) > 0.8",
                  "rank(vec_avg(mws85_sentiment)) > 0.8", "ts_rank(vec_avg(mws85_sentiment),22) > 0.8",
                  "rank(mdl110_analyst_sentiment) > 0.8", "ts_rank(mdl110_analyst_sentiment, 22) > 0.8",
                  "rank(vec_avg(nws3_scores_posnormscr)) > 0.8",
                  "ts_rank(vec_avg(nws3_scores_posnormscr),22) > 0.8",
                  "rank(vec_avg(mws36_sentiment_words_positive)) > 0.8",
                  "ts_rank(vec_avg(mws36_sentiment_words_positive),22) > 0.8"]

    glb_events = ["rank(vec_avg(mdl109_news_sent_1m)) > 0.8",
                  "ts_rank(vec_avg(mdl109_news_sent_1m),22) > 0.8",
                  "rank(vec_avg(nws20_ssc)) > 0.8",
                  "ts_rank(vec_avg(nws20_ssc),22) > 0.8",
                  "vec_avg(nws20_ssc) > 0",
                  "rank(vec_avg(nws20_bee)) > 0.8",
                  "ts_rank(vec_avg(nws20_bee),22) > 0.8",
                  "rank(vec_avg(nws20_qmb)) > 0.8",
                  "ts_rank(vec_avg(nws20_qmb),22) > 0.8"]

    chn_events = ["rank(vec_avg(oth111_xueqiunaturaldaybasicdivisionstat_senti_conform)) > 0.8",
                  "ts_rank(vec_avg(oth111_xueqiunaturaldaybasicdivisionstat_senti_conform),22) > 0.8",
                  "rank(vec_avg(oth111_gubanaturaldaydevicedivisionstat_senti_conform)) > 0.8",
                  "ts_rank(vec_avg(oth111_gubanaturaldaydevicedivisionstat_senti_conform),22) > 0.8",
                  "rank(vec_avg(oth111_baragedivisionstat_regi_senti_conform)) > 0.8",
                  "ts_rank(vec_avg(oth111_baragedivisionstat_regi_senti_conform),22) > 0.8"]

    kor_events = ["rank(vec_avg(mdl110_analyst_sentiment)) > 0.8",
                  "ts_rank(vec_avg(mdl110_analyst_sentiment),22) > 0.8",
                  "rank(vec_avg(mws38_score)) > 0.8",
                  "ts_rank(vec_avg(mws38_score),22) > 0.8"]

    twn_events = ["rank(vec_avg(mdl109_news_sent_1m)) > 0.8",
                  "ts_rank(vec_avg(mdl109_news_sent_1m),22) > 0.8",
                  "rank(rp_ess_business) > 0.8",
                  "ts_rank(rp_ess_business,22) > 0.8"]

    regional_events = {
        "USA": usa_events,
        "ASI": asi_events,
        "EUR": eur_events,
        "GLB": glb_events,
        "CHN": chn_events,
        "KOR": kor_events,
        "TWN": twn_events,
    }
    open_events += regional_events.get(region.upper(), [])

    for oe in open_events:
        for ee in exit_events:
            alpha = "%s(%s, %s, %s)"%(op, oe, field, ee)
            output.append(alpha)
    return output


def check_submission(alpha_bag, gold_bag, start, max_retries=2):
    depot = []
    s = login()
    pending = deque((idx, alpha_id, 0) for idx, alpha_id in enumerate(alpha_bag))
    processed = 0
    while pending:
        idx, g, retries = pending.popleft()
        if idx < start:
            continue
        if idx % 5 == 0:
            print(idx)
        if processed and processed % 200 == 0:
            s = login()
        processed += 1
        pc = get_check_submission(s, g)
        should_retry = pc == "sleep" or (
            isinstance(pc, float) and math.isnan(pc)
        )
        if should_retry and retries < max_retries:
            sleep(100)
            s = login()
            pending.append((idx, g, retries + 1))
        elif should_retry:
            depot.append(g)
        elif pc == "fail":
            continue
        elif pc == "error":
            depot.append(g)
        else:
            print(g)
            gold_bag.append((g, pc))
    print(depot)
    return gold_bag


def get_check_submission(s, alpha_id):
    while True:
        result = s.get(
            f"{API_BASE_URL}/alphas/{alpha_id}/check",
            timeout=DEFAULT_TIMEOUT,
        )
        retry_after = result.headers.get("Retry-After")
        if retry_after:
            time.sleep(float(retry_after))
        else:
            break
    try:
        result.raise_for_status()
        payload = result.json()
        if not payload.get("is"):
            print("logged out")
            return "sleep"
        checks_df = pd.DataFrame(payload["is"]["checks"])
        self_correlation = checks_df[checks_df.name == "SELF_CORRELATION"]
        if self_correlation.empty:
            return "error"
        pc = self_correlation["value"].iloc[0]
        if not any(checks_df["result"] == "FAIL"):
            return pc
        return "fail"
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        print(f"check failed for {alpha_id}: {exc}")
        return "error"
    
def view_alphas(gold_bag):
    s = login()
    sharp_list = []
    for gold, pc in gold_bag:

        triple = locate_alpha(s, gold)
        info = [triple[0], triple[2], triple[3], triple[4], triple[5], triple[6], triple[1]]
        info.append(pc)
        sharp_list.append(info)

    sharp_list.sort(reverse=True, key = lambda x : x[1])
    for i in sharp_list:
        print(i)
 
def locate_alpha(s, alpha_id):
    while True:
        alpha = s.get(
            f"{API_BASE_URL}/alphas/{alpha_id}",
            timeout=DEFAULT_TIMEOUT,
        )
        retry_after = alpha.headers.get("Retry-After")
        if retry_after:
            time.sleep(float(retry_after))
        else:
            break
    alpha.raise_for_status()
    metrics = alpha.json()
    
    dateCreated = metrics["dateCreated"]
    sharpe = metrics["is"]["sharpe"]
    fitness = metrics["is"]["fitness"]
    turnover = metrics["is"]["turnover"]
    margin = metrics["is"]["margin"]
    decay = metrics["settings"]["decay"]
    exp = metrics['regular']['code']
    
    triple = [alpha_id, exp, sharpe, turnover, fitness, margin, dateCreated, decay]
    return triple
            

# Consultant methods
def multi_simulate(alpha_pools, neut, region, universe, start):
    s = login()
    failures = []

    for x, pool in enumerate(alpha_pools):
        if x < start:
            continue
        progress_urls = []
        for y, task in enumerate(pool):
            sim_data_list = generate_sim_data(task, region, universe, neut)
            try:
                response = s.post(
                    f"{API_BASE_URL}/simulations",
                    json=sim_data_list,
                    timeout=DEFAULT_TIMEOUT,
                )
                response.raise_for_status()
                progress_url = response.headers.get("Location")
                if not progress_url:
                    raise KeyError("Missing Location header")
                progress_urls.append((y, progress_url))
            except (requests.RequestException, KeyError) as exc:
                failures.append((x, y, str(exc).strip("'")))
                print(f"pool {x} task {y} post failed: {exc}")

        print("pool %d posted %d tasks"%(x, len(pool)))

        for task_index, progress in progress_urls:
            try:
                while True:
                    simulation_progress = s.get(
                        progress,
                        timeout=DEFAULT_TIMEOUT,
                    )
                    retry_after = simulation_progress.headers.get("Retry-After")
                    if not retry_after:
                        break
                    sleep(float(retry_after))

                simulation_progress.raise_for_status()
                status = simulation_progress.json().get("status", 0)
                if status not in {"COMPLETE", "WARNING"}:
                    print("Not complete : %s"%(progress))
                    failures.append((x, task_index, f"status={status}"))
            except (requests.RequestException, ValueError, TypeError) as exc:
                failures.append((x, task_index, str(exc)))
                print(f"simulation check failed for {progress}: {exc}")

        print("pool %d simulated %d tasks"%(x, len(progress_urls)))
    
    print("Simulate done")
    return failures

def generate_sim_data(alpha_list, region, uni, neut):
    sim_data_list = []
    for alpha, decay in alpha_list:
        simulation_data = {
            'type': 'REGULAR',
            'settings': {
                'instrumentType': 'EQUITY',
                'region': region,
                'universe': uni,
                'delay': 1,
                'decay': decay,
                'neutralization': neut,
                'truncation': 0.08,
                'pasteurization': 'ON',
                'testPeriod': 'P2Y',
                'unitHandling': 'VERIFY',
                'nanHandling': 'ON',
                'language': 'FASTEXPR',
                'visualization': False,
            },
            'regular': alpha}

        sim_data_list.append(simulation_data)
    return sim_data_list

def load_task_pool(alpha_list, limit_of_children_simulations, limit_of_multi_simulations):
    '''
    Input:
        alpha_list : list of (alpha, decay) tuples
        limit_of_multi_simulations : number of children simulation in a multi-simulation
        limit_of_multi_simulations : number of simultaneous multi-simulations
    Output:
        task : [10 * (alpha, decay)] for a multi-simulation
        pool : [10 * [10 * (alpha, decay)]] for simultaneous multi-simulations
        pools : [[10 * [10 * (alpha, decay)]]]

    '''
    if limit_of_children_simulations <= 0 or limit_of_multi_simulations <= 0:
        raise ValueError("simulation limits must be positive")
    tasks = [
        alpha_list[i:i + limit_of_children_simulations]
        for i in range(0, len(alpha_list), limit_of_children_simulations)
    ]
    pools = [
        tasks[i:i + limit_of_multi_simulations]
        for i in range(0, len(tasks), limit_of_multi_simulations)
    ]
    return pools


# some other factory for other operators
def vector_factory(op, field):
    output = []
    vectors = ["cap"]
    
    for vector in vectors:
    
        alpha = "%s(%s, %s)"%(op, field, vector)
        output.append(alpha)
    
    return output
 
def ts_comp_factory(op, field, factor, paras):
    output = []
    #l1, l2 = [3, 5, 10, 20, 60, 120, 240], paras
    l1, l2 = [5, 22, 66, 240], paras
    comb = list(product(l1, l2))
    
    for day,para in comb:
        
        if isinstance(para, float):
            alpha = "%s(%s, %d, %s=%.1f)"%(op, field, day, factor, para)
        elif isinstance(para, int) and not isinstance(para, bool):
            alpha = "%s(%s, %d, %s=%d)"%(op, field, day, factor, para)
        else:
            raise TypeError("operator parameters must be int or float")
        output.append(alpha)
    
    return output
 
def twin_field_factory(op, field, fields):
    
    output = []
    #days = [3, 5, 10, 20, 60, 120, 240]
    days = [5, 22, 66, 240]
    outset = list(set(fields) - set([field]))
    
    for day in days:
        for counterpart in outset:
            alpha = "%s(%s, %s, %d)"%(op, field, counterpart, day)
            output.append(alpha)
    
    return output
 
def login_hk():
    """Authenticate, including the optional Persona biometric challenge."""
    username, password = load_wqb_credentials()

    s = requests.Session()
    s.auth = (username, password)
    response = s.post(
        f"{API_BASE_URL}/authentication",
        timeout=DEFAULT_TIMEOUT,
    )
    
    if response.status_code == requests.codes.unauthorized:
        if response.headers.get("WWW-Authenticate") == "persona":
            location = response.headers.get("Location")
            if not location:
                raise RuntimeError("Biometric authentication response has no Location")
            biometric_url = urljoin(response.url, location)
            print(
                "Complete biometrics authentication by scanning your face. Follow the link: \n"
                + biometric_url + "\n"
            )
            input("Press any key after you complete the biometrics authentication.")

            biometrics_response = s.post(
                biometric_url,
                timeout=DEFAULT_TIMEOUT,
            )
            while biometrics_response.status_code != 201:
                if biometrics_response.status_code != requests.codes.unauthorized:
                    biometrics_response.raise_for_status()
                input("Biometrics authentication is not complete. Please try again and press any key when completed.")
                biometrics_response = s.post(
                    biometric_url,
                    timeout=DEFAULT_TIMEOUT,
                )
            print("Biometrics authentication completed.")
        else:
            response.raise_for_status()
    else:
        response.raise_for_status()
        print("Logged in successfully.")
    
    return s
