from __future__ import annotations

import json
import math
import sqlite3

import pytest


def create_well(client,code="W-001"):
    response=client.post("/api/hydro/wells",json={"code":code,"name":"北部监测井","latitude":35.1,"longitude":116.2,"aquifer":"浅层孔隙含水层","screen_depth_m":42})
    assert response.status_code==201,response.text
    return response.json()


def create_endmember(client,name,d18o,d2h,solute,uncertainty=0.1):
    response=client.post("/api/hydro/endmembers",json={"name":name,"isotope_d18o":d18o,"isotope_d2h":d2h,"solute_mg_l":solute,"uncertainty":uncertainty,"version":"v1"})
    assert response.status_code==201,response.text
    return response.json()


def add_sample(client,well_id,code,**fields):
    payload={"sample_code":code,"sampled_at":"2026-09-24T08:00:00+00:00","detection_limit":0.1,"measurement_error":0.05}
    payload.update(fields)
    response=client.post(f"/api/hydro/wells/{well_id}/samples",json=payload)
    assert response.status_code==201,response.text
    return response.json()


def run_inversion(client,sample_id,endmember_ids,**overrides):
    payload={"endmember_ids":endmember_ids,"max_iterations":2000,"tolerance":1e-12,"model_version":"mix-test"}
    payload.update(overrides)
    task=client.post(f"/api/hydro/samples/{sample_id}/inversions",json=payload)
    assert task.status_code==202,task.text
    done=client.post(f"/api/hydro/inversions/{task.json()['id']}/run?worker_id=test")
    assert done.status_code==200,done.text
    assert done.json()["status"]=="done"
    return json.loads(done.json()["result_json"])


def fractions_of(result):
    return [f["fraction"] for f in result["fractions"]]


def test_mixture_inversion_and_transport(client):
    well=create_well(client)
    e1=client.post("/api/hydro/endmembers",json={"name":"山区降水","isotope_d18o":-10,"isotope_d2h":-70,"solute_mg_l":10,"uncertainty":0.1,"version":"v1"}).json()
    e2=client.post("/api/hydro/endmembers",json={"name":"河流渗漏","isotope_d18o":-5,"isotope_d2h":-35,"solute_mg_l":50,"uncertainty":0.2,"version":"v1"}).json()
    sample=client.post(f"/api/hydro/wells/{well['id']}/samples",json={"sample_code":"S-001","sampled_at":"2026-09-24T08:00:00+00:00","isotope_d18o":-7.5,"isotope_d2h":-52.5,"solute_mg_l":30,"detection_limit":0.1,"measurement_error":0.05}).json()
    task=client.post(f"/api/hydro/samples/{sample['id']}/inversions",json={"endmember_ids":[e1['id'],e2['id']],"max_iterations":1000,"tolerance":1e-10,"model_version":"mix-test"})
    assert task.status_code==202,task.text
    done=client.post(f"/api/hydro/inversions/{task.json()['id']}/run?worker_id=test")
    assert done.status_code==200,done.text
    assert done.json()["status"]=="done"
    transport=client.post(f"/api/hydro/wells/{well['id']}/transport",json={"source_concentration":100,"distance_m":100,"velocity_m_day":2,"dispersion_m2_day":5,"decay_per_day":0.01,"duration_days":100,"step_days":5,"model_version":"ade-test"})
    assert transport.status_code==201,transport.text
    assert transport.json()["result_json"]


def test_missing_measurement_is_classified(client):
    well=create_well(client,"W-002")
    sample=client.post(f"/api/hydro/wells/{well['id']}/samples",json={"sample_code":"S-002","sampled_at":"2026-09-24T08:00:00+00:00","isotope_d18o":-7.5,"detection_limit":0.1,"measurement_error":0.05})
    assert sample.status_code==201
    assert sample.json()["quality_status"]=="incomplete"


def test_censored_observation_ignores_reported_value(client):
    well=create_well(client,"W-101")
    e1=create_endmember(client,"山区降水",-10,-70,10)
    e2=create_endmember(client,"河流渗漏",-5,-35,50)
    ids=[e1["id"],e2["id"]]
    def censored_sample(code,value):
        fields={"isotope_d18o":-7.5,"isotope_d2h":-52.5,"observations":{"solute_mg_l":{"type":"censored","detection_limit":25.0}}}
        if value is not None: fields["solute_mg_l"]=value
        return add_sample(client,well["id"],code,**fields)
    # 同一删失观测，实验室报告值不同（甚至缺失），反演结果必须完全一致：删失值只用检出限
    low=censored_sample("SC-001",0.05)
    high=censored_sample("SC-002",20.0)
    unreported=censored_sample("SC-003",None)
    result_low=run_inversion(client,low["id"],ids)
    assert result_low["status"]=="solved"
    assert fractions_of(result_low)==fractions_of(run_inversion(client,high["id"],ids))
    assert fractions_of(result_low)==fractions_of(run_inversion(client,unreported["id"],ids))
    solute_entry=next(o for o in result_low["observations"] if o["tracer"]=="solute_mg_l")
    assert solute_entry["role"]=="censored_likelihood"
    assert solute_entry["detection_limit"]==25.0
    assert solute_entry["log_likelihood"]<=0
    # 同一数值若按定量处理则必须作为残差参与，角色与结果都不同
    quantitative=add_sample(client,well["id"],"SC-004",isotope_d18o=-7.5,isotope_d2h=-52.5,solute_mg_l=25.0)
    result_quant=run_inversion(client,quantitative["id"],ids)
    solute_quant=next(o for o in result_quant["observations"] if o["tracer"]=="solute_mg_l")
    assert solute_quant["role"]=="residual"
    assert fractions_of(result_quant)!=fractions_of(result_low)


def test_all_censored_or_missing_is_unsolvable(client):
    well=create_well(client,"W-102")
    e1=create_endmember(client,"山区降水",-10,-70,10)
    e2=create_endmember(client,"河流渗漏",-5,-35,50)
    ids=[e1["id"],e2["id"]]
    # 样品量不足只测了溶质且低于检出限：不允许生成看似精确的比例
    censored_only=add_sample(client,well["id"],"SU-001",observations={"solute_mg_l":{"type":"censored","detection_limit":0.1}})
    result=run_inversion(client,censored_only["id"],ids)
    assert result["status"]=="unsolvable"
    assert result["reason"]=="no_quantitative_measurements"
    assert result["fractions"] is None
    assert "删失" in result["reason_detail"] or "检出限" in result["reason_detail"]
    roles={o["tracer"]:o["role"] for o in result["observations"]}
    assert roles=={"isotope_d18o":"excluded","isotope_d2h":"excluded","solute_mg_l":"censored_likelihood"}
    # 全部缺测同样不可求解
    empty=add_sample(client,well["id"],"SU-002")
    result_empty=run_inversion(client,empty["id"],ids)
    assert result_empty["status"]=="unsolvable"
    assert result_empty["reason"]=="no_quantitative_measurements"
    assert result_empty["fractions"] is None


def test_underdetermined_system_is_unsolvable(client):
    well=create_well(client,"W-103")
    e1=create_endmember(client,"山区降水",-10,-70,10)
    e2=create_endmember(client,"河流渗漏",-5,-35,50)
    e3=create_endmember(client,"深层地下水",-12,-80,200)
    ids=[e1["id"],e2["id"],e3["id"]]
    # 3 个端元只有 1 个定量示踪剂：欠定，必须给出原因而不是比例
    sparse=add_sample(client,well["id"],"SU-101",isotope_d18o=-8.0)
    result=run_inversion(client,sparse["id"],ids)
    assert result["status"]=="unsolvable"
    assert result["reason"]=="underdetermined_system"
    assert result["fractions"] is None
    assert result["quantitative_tracers"]==1
    assert result["required_quantitative_tracers"]==2
    # 边界：定量示踪剂数恰好等于端元数-1 时可解
    exactly=add_sample(client,well["id"],"SU-102",isotope_d18o=-8.0,isotope_d2h=-55.0)
    result_exactly=run_inversion(client,exactly["id"],ids)
    assert result_exactly["status"]=="solved"
    assert abs(sum(fractions_of(result_exactly))-1.0)<1e-6


def test_uninformative_tracer_does_not_count(client):
    well=create_well(client,"W-104")
    # 两个端元的 δ18O 完全相同，该示踪剂对混合比例没有区分能力
    e1=create_endmember(client,"端元甲",-8,-70,10)
    e2=create_endmember(client,"端元乙",-8,-35,50)
    ids=[e1["id"],e2["id"]]
    flat=add_sample(client,well["id"],"SU-201",isotope_d18o=-8.0)
    result=run_inversion(client,flat["id"],ids)
    assert result["status"]=="unsolvable"
    assert result["reason"]=="underdetermined_system"
    assert result["informative_quantitative_tracers"]==0
    # 换成有区分度的 δ2H 即可求解
    informative=add_sample(client,well["id"],"SU-202",isotope_d2h=-52.5)
    result_ok=run_inversion(client,informative["id"],ids)
    assert result_ok["status"]=="solved"
    assert fractions_of(result_ok)==[0.5,0.5]


def test_legacy_sample_keeps_reported_values(client):
    well=create_well(client,"W-105")
    e1=create_endmember(client,"山区降水",-10,-70,10)
    e2=create_endmember(client,"河流渗漏",-5,-35,50)
    # 历史写法：不声明观测类型，只填报告值
    sample=add_sample(client,well["id"],"SL-001",isotope_d18o=-7.5,solute_mg_l=30)
    stored=json.loads(sample["observations_json"])
    assert stored["isotope_d18o"]["type"]=="quantitative"
    assert stored["isotope_d2h"]["type"]=="missing"
    assert stored["solute_mg_l"]["type"]=="quantitative"
    fetched=client.get(f"/api/hydro/wells/{well['id']}").json()["samples"][0]
    assert fetched["isotope_d18o"]==-7.5
    assert fetched["isotope_d2h"] is None
    assert fetched["solute_mg_l"]==30
    result=run_inversion(client,sample["id"],[e1["id"],e2["id"]])
    assert result["status"]=="solved"
    roles={o["tracer"]:(o["type"],o["role"]) for o in result["observations"]}
    assert roles["isotope_d18o"]==("quantitative","residual")
    assert roles["solute_mg_l"]==("quantitative","residual")
    assert roles["isotope_d2h"]==("missing","excluded")
    values={o["tracer"]:o.get("value") for o in result["observations"]}
    assert values["isotope_d18o"]==-7.5 and values["solute_mg_l"]==30


def test_observation_type_validation(client):
    well=create_well(client,"W-106")
    def post(code,**fields):
        payload={"sample_code":code,"sampled_at":"2026-09-24T08:00:00+00:00","detection_limit":0.1,"measurement_error":0.05}
        payload.update(fields)
        return client.post(f"/api/hydro/wells/{well['id']}/samples",json=payload)
    # 定量声明却没有数值
    assert post("SV-001",observations={"isotope_d18o":{"type":"quantitative"}}).status_code==422
    # 缺失声明却带了数值
    assert post("SV-002",isotope_d18o=-7.5,observations={"isotope_d18o":{"type":"missing"}}).status_code==422
    # 删失声明却没有任何正的检出限
    assert post("SV-003",detection_limit=0,observations={"solute_mg_l":{"type":"censored"}}).status_code==422
    # 未知示踪剂名与未知类型
    assert post("SV-004",observations={"isotope_d15n":{"type":"censored","detection_limit":0.1}}).status_code==422
    assert post("SV-005",observations={"solute_mg_l":{"type":"estimated","detection_limit":0.1}}).status_code==422
    # 删失可以使用样本级检出限，也可以用逐项覆盖
    assert post("SV-006",observations={"solute_mg_l":{"type":"censored"}}).status_code==201
    assert post("SV-007",detection_limit=0,observations={"solute_mg_l":{"type":"censored","detection_limit":0.2}}).status_code==201


def test_quality_status_counts_censored_as_measured(client):
    well=create_well(client,"W-107")
    sample=add_sample(client,well["id"],"SQ-001",isotope_d18o=-7.5,observations={"solute_mg_l":{"type":"censored","detection_limit":0.1}})
    assert sample["quality_status"]=="usable"
    sparse=add_sample(client,well["id"],"SQ-002",isotope_d18o=-7.5)
    assert sparse["quality_status"]=="incomplete"


def test_result_lists_observation_participation(client):
    well=create_well(client,"W-108")
    e1=create_endmember(client,"山区降水",-10,-70,10)
    e2=create_endmember(client,"河流渗漏",-5,-35,50)
    sample=add_sample(client,well["id"],"SP-001",isotope_d18o=-7.5,isotope_d2h=-52.5,observations={"solute_mg_l":{"type":"censored","detection_limit":25.0}})
    result=run_inversion(client,sample["id"],[e1["id"],e2["id"]])
    assert result["status"]=="solved"
    assert result["reason"] is None
    assert result["quantitative_tracers"]==2
    assert result["censored_tracers"]==1
    assert result["missing_tracers"]==0
    by_tracer={o["tracer"]:o for o in result["observations"]}
    assert by_tracer["isotope_d18o"]["role"]=="residual"
    assert "residual" in by_tracer["isotope_d18o"] and "weight" in by_tracer["isotope_d18o"]
    assert by_tracer["solute_mg_l"]["role"]=="censored_likelihood"
    assert by_tracer["solute_mg_l"]["detection_limit"]==25.0
    assert "z_score" in by_tracer["solute_mg_l"]
    assert result["censored_log_likelihood"]<=0
    # 删失约束应把比例推向低溶质端元（无约束时同位素恰好给出 0.5/0.5）
    assert fractions_of(result)[0]>0.5
    assert result["predicted"][2]<30.0


def test_censored_likelihood_is_monotone_in_detection_limit(client):
    well=create_well(client,"W-109")
    e1=create_endmember(client,"山区降水",-10,-70,10)
    e2=create_endmember(client,"河流渗漏",-5,-35,50)
    ids=[e1["id"],e2["id"]]
    solved={}
    for i,dl in enumerate([1000.0,29.0,15.0]):
        sample=add_sample(client,well["id"],f"SM-{i:03d}",isotope_d18o=-7.5,isotope_d2h=-52.5,observations={"solute_mg_l":{"type":"censored","detection_limit":dl}})
        solved[dl]=run_inversion(client,sample["id"],ids)
    # 检出限远高于可达预测值时删失约束不起作用，退化为仅用同位素的结果
    assert abs(fractions_of(solved[1000.0])[0]-0.5)<1e-6
    # 检出限越低，删失似然把解推向低溶质端元的力度越强
    f_high,f_mid,f_low=(fractions_of(solved[dl])[0] for dl in (1000.0,29.0,15.0))
    assert f_high<=f_mid<=f_low
    assert f_mid>f_high
    predicted_solute=[solved[dl]["predicted"][2] for dl in (1000.0,29.0,15.0)]
    assert predicted_solute[0]>=predicted_solute[1]>=predicted_solute[2]


def test_solver_boundary_numerics(client):
    from app.hydro.service import HydroService, _log_norm_cdf, _mills_ratio
    # 对数正态累积与米尔斯比在极端输入下保持有限且渐近正确
    assert _log_norm_cdf(0.0)==math.log(0.5)
    assert math.isfinite(_log_norm_cdf(-1e6))
    assert _log_norm_cdf(-1e6)==pytest.approx(-0.5e12,rel=1e-6)
    assert _mills_ratio(0.0)==pytest.approx(math.sqrt(2.0/math.pi),rel=1e-12)
    assert _mills_ratio(-1e6)==pytest.approx(1e6,rel=1e-6)
    assert 0.0<=_mills_ratio(30.0)<1e-100
    service=HydroService()
    endmembers=[{"id":1,"name":"e1","isotope_d18o":-10.0,"isotope_d2h":-70.0,"solute_mg_l":10.0},{"id":2,"name":"e2","isotope_d18o":-5.0,"isotope_d2h":-35.0,"solute_mg_l":50.0}]
    base={"isotope_d18o":-7.5,"isotope_d2h":-52.5,"solute_mg_l":None,"detection_limit":0.1,"measurement_error":0.05,"observations_json":""}
    solved=service.solve_mixture(base,endmembers,2000,1e-12)
    assert solved["status"]=="solved"
    assert abs(solved["fractions"][0]["fraction"]-0.5)<1e-6
    # 检出限极高 → 删失项完全不起作用，结果与缺测完全一致
    inactive={**base,"observations_json":json.dumps({"solute_mg_l":{"type":"censored","detection_limit":1000.0}})}
    solved_inactive=service.solve_mixture(inactive,endmembers,2000,1e-12)
    assert [f["fraction"] for f in solved_inactive["fractions"]]==[f["fraction"] for f in solved["fractions"]]
    # 无定量值时求解器直接返回不可求解，不产生比例
    censored_only={**base,"isotope_d18o":None,"isotope_d2h":None,"observations_json":json.dumps({"solute_mg_l":{"type":"censored","detection_limit":0.1}})}
    refused=service.solve_mixture(censored_only,endmembers,2000,1e-12)
    assert refused["status"]=="unsolvable"
    assert refused["reason"]=="no_quantitative_measurements"
    assert refused["fractions"] is None


def test_schema_migration_preserves_legacy_rows(tmp_path,monkeypatch):
    import app.database as database
    from app.hydro.service import ensure_schema
    db_path=tmp_path/"legacy.db"
    legacy=sqlite3.connect(db_path)
    legacy.execute("CREATE TABLE hydro_samples (id INTEGER PRIMARY KEY AUTOINCREMENT, well_id INTEGER NOT NULL, sample_code TEXT NOT NULL UNIQUE, sampled_at TEXT NOT NULL, isotope_d18o REAL, isotope_d2h REAL, solute_mg_l REAL, detection_limit REAL NOT NULL, measurement_error REAL NOT NULL, quality_status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL)")
    legacy.execute("INSERT INTO hydro_samples(well_id,sample_code,sampled_at,isotope_d18o,isotope_d2h,solute_mg_l,detection_limit,measurement_error,quality_status,created_at) VALUES(1,'S-OLD','2020-01-01T00:00:00+00:00',-7.5,NULL,30,0.1,0.05,'usable','2020-01-01T00:00:00+00:00')")
    legacy.commit()
    legacy.close()
    monkeypatch.setenv("TOWNSHIP_DATABASE_PATH",str(db_path))
    database.close_connection()
    ensure_schema()
    check=sqlite3.connect(db_path)
    columns={row[1] for row in check.execute("PRAGMA table_info(hydro_samples)")}
    assert "observations_json" in columns
    row=check.execute("SELECT isotope_d18o,isotope_d2h,solute_mg_l,observations_json FROM hydro_samples WHERE sample_code='S-OLD'").fetchone()
    assert row[0]==-7.5 and row[1] is None and row[2]==30
    assert row[3]==""
    check.close()
    database.close_connection()
