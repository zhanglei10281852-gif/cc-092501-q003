from __future__ import annotations

import json
import sqlite3

import pytest


def create_well(client, code):
    response = client.post("/api/hydro/wells", json={"code": code, "name": "观测井", "latitude": 35.1, "longitude": 116.2, "aquifer": "浅层孔隙含水层", "screen_depth_m": 42})
    assert response.status_code == 201, response.text
    return response.json()


def create_endmembers(client):
    e1 = client.post("/api/hydro/endmembers", json={"name": "山区降水", "isotope_d18o": -10, "isotope_d2h": -70, "solute_mg_l": 10, "uncertainty": 0.1, "version": "v1"}).json()
    e2 = client.post("/api/hydro/endmembers", json={"name": "河流渗漏", "isotope_d18o": -5, "isotope_d2h": -35, "solute_mg_l": 50, "uncertainty": 0.2, "version": "v1"}).json()
    return [e1["id"], e2["id"]]


def add_sample(client, well_id, code, **fields):
    payload = {"sample_code": code, "sampled_at": "2026-09-24T08:00:00+00:00", "detection_limit": 0.1, "measurement_error": 0.05}
    payload.update(fields)
    response = client.post(f"/api/hydro/wells/{well_id}/samples", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def run_inversion(client, sample_id, endmember_ids):
    task = client.post(f"/api/hydro/samples/{sample_id}/inversions", json={"endmember_ids": endmember_ids, "max_iterations": 2000, "tolerance": 1e-12, "model_version": "mix-obs"})
    assert task.status_code == 202, task.text
    done = client.post(f"/api/hydro/inversions/{task.json()['id']}/run?worker_id=test")
    assert done.status_code == 200, done.text
    assert done.json()["status"] == "done"
    return json.loads(done.json()["result_json"])


def fractions_of(result):
    return {entry["name"]: entry["fraction"] for entry in result["fractions"]}


def observation_of(result, channel):
    return next(entry for entry in result["observations"] if entry["channel"] == channel)


# 端元同位素特征为 -10/-70 与 -5/-35，样本 -7.5/-52.5 恰好对应 0.5/0.5 混合，
# 此时溶质预测值恒为 0.5*10 + 0.5*50 = 30，便于精确复现删失边界。

def test_censored_satisfied_behaves_like_missing_channel(client):
    well = create_well(client, "W-C1")
    ids = create_endmembers(client)
    missing = add_sample(client, well["id"], "S-C1-M", isotope_d18o=-7.5, isotope_d2h=-52.5)
    censored = add_sample(client, well["id"], "S-C1-C", isotope_d18o=-7.5, isotope_d2h=-52.5, solute_mg_l=100.0, observations={"solute_mg_l": {"type": "censored"}})
    result_missing = run_inversion(client, missing["id"], ids)
    result_censored = run_inversion(client, censored["id"], ids)
    # 检出限高于任何可行预测：约束恒满足，删失通道与完全缺失产生相同结果
    assert fractions_of(result_censored) == pytest.approx(fractions_of(result_missing), abs=1e-9)
    assert fractions_of(result_censored)["山区降水"] == pytest.approx(0.5, abs=1e-6)
    entry = observation_of(result_censored, "solute_mg_l")
    assert entry["participation"] == "one-sided-inequality"
    assert entry["satisfied"] is True
    assert entry["residual"] == 0.0
    assert entry["contribution"] == 0.0


def test_censored_value_is_not_treated_as_exact_number(client):
    well = create_well(client, "W-C2")
    ids = create_endmembers(client)
    quantitative = add_sample(client, well["id"], "S-C2-Q", isotope_d18o=-7.5, isotope_d2h=-52.5, solute_mg_l=50.0)
    censored = add_sample(client, well["id"], "S-C2-C", isotope_d18o=-7.5, isotope_d2h=-52.5, solute_mg_l=50.0, observations={"solute_mg_l": {"type": "censored"}})
    result_quantitative = run_inversion(client, quantitative["id"], ids)
    result_censored = run_inversion(client, censored["id"], ids)
    fractions_quantitative = fractions_of(result_quantitative)
    fractions_censored = fractions_of(result_censored)
    # 同样的数值 50：定量观测把比例拉向高溶质端元，删失观测因约束满足而不施加拉力
    assert fractions_quantitative["河流渗漏"] > 0.6
    assert fractions_censored["河流渗漏"] == pytest.approx(0.5, abs=1e-6)
    assert abs(fractions_quantitative["河流渗漏"] - fractions_censored["河流渗漏"]) > 0.1


def test_censored_boundary_exactly_at_limit(client):
    well = create_well(client, "W-C3")
    ids = create_endmembers(client)
    missing = add_sample(client, well["id"], "S-C3-M", isotope_d18o=-7.5, isotope_d2h=-52.5)
    at_limit = add_sample(client, well["id"], "S-C3-K", isotope_d18o=-7.5, isotope_d2h=-52.5, solute_mg_l=30.0, observations={"solute_mg_l": {"type": "censored"}})
    result_missing = run_inversion(client, missing["id"], ids)
    result_kink = run_inversion(client, at_limit["id"], ids)
    # 预测值恰好等于检出限（铰链点）：删失残差为 0，结果与该通道缺失时完全一致
    assert fractions_of(result_kink) == pytest.approx(fractions_of(result_missing), abs=1e-9)
    entry = observation_of(result_kink, "solute_mg_l")
    assert entry["bound"] == 30.0
    assert entry["predicted"] == pytest.approx(30.0, abs=1e-9)
    assert entry["residual"] == 0.0
    assert entry["contribution"] == 0.0
    assert entry["satisfied"] is True


def test_censored_violated_constraint_pulls_fractions(client):
    well = create_well(client, "W-C4")
    ids = create_endmembers(client)
    missing = add_sample(client, well["id"], "S-C4-M", isotope_d18o=-7.5, isotope_d2h=-52.5)
    # 检出限 25 略低于同位素隐含的预测 30：内点解，单侧残差保持为正值
    violated = add_sample(client, well["id"], "S-C4-V", isotope_d18o=-7.5, isotope_d2h=-52.5, solute_mg_l=25.0, observations={"solute_mg_l": {"type": "censored"}})
    result_missing = run_inversion(client, missing["id"], ids)
    result_violated = run_inversion(client, violated["id"], ids)
    fractions = fractions_of(result_violated)
    assert fractions["山区降水"] > 0.55
    assert fractions["山区降水"] > fractions_of(result_missing)["山区降水"]
    entry = observation_of(result_violated, "solute_mg_l")
    assert entry["satisfied"] is False
    assert entry["residual"] > 0
    assert entry["contribution"] > 0


def test_censored_strongly_violated_drives_prediction_to_limit(client):
    well = create_well(client, "W-C5")
    ids = create_endmembers(client)
    # 检出限 10 远低于同位素隐含的预测 30：比例被压到低溶质端元顶点，预测恰好停在检出限上
    violated = add_sample(client, well["id"], "S-C5-V", isotope_d18o=-7.5, isotope_d2h=-52.5, solute_mg_l=10.0, observations={"solute_mg_l": {"type": "censored"}})
    result = run_inversion(client, violated["id"], ids)
    assert fractions_of(result)["山区降水"] == pytest.approx(1.0)
    entry = observation_of(result, "solute_mg_l")
    assert entry["bound"] == 10.0
    assert entry["predicted"] <= entry["bound"] + 1e-9


def test_unsolvable_reasons_are_explicit(client):
    well = create_well(client, "W-U1")
    ids = create_endmembers(client)

    all_missing = add_sample(client, well["id"], "S-U1-AM")
    response = client.post(f"/api/hydro/samples/{all_missing['id']}/inversions", json={"endmember_ids": ids})
    assert response.status_code == 422
    assert "unsolvable:no_quantitative_measurements" in response.json()["detail"]

    censored_only = add_sample(client, well["id"], "S-U1-CO", solute_mg_l=0.1, observations={"solute_mg_l": {"type": "censored"}})
    response = client.post(f"/api/hydro/samples/{censored_only['id']}/inversions", json={"endmember_ids": ids})
    assert response.status_code == 422
    assert "unsolvable:no_quantitative_measurements" in response.json()["detail"]

    single_quantitative = add_sample(client, well["id"], "S-U1-1Q", isotope_d18o=-7.5)
    response = client.post(f"/api/hydro/samples/{single_quantitative['id']}/inversions", json={"endmember_ids": ids})
    assert response.status_code == 422
    assert "unsolvable:insufficient_measurements" in response.json()["detail"]

    # 可求解性边界：1 项定量 + 1 项左删失即可求解
    boundary = add_sample(client, well["id"], "S-U1-B", isotope_d18o=-7.5, solute_mg_l=100.0, observations={"solute_mg_l": {"type": "censored"}})
    result = run_inversion(client, boundary["id"], ids)
    assert result["mass_balance"] == pytest.approx(1.0)


def test_solver_raises_unsolvable_for_censored_only_sample(client):
    from app.hydro.service import HydroService, UnsolvableError

    service = HydroService()
    sample = {"isotope_d18o": None, "isotope_d2h": None, "solute_mg_l": 0.1, "observation_types": json.dumps({"solute_mg_l": "censored"})}
    endmembers = [
        {"id": 1, "name": "e1", "isotope_d18o": -10.0, "isotope_d2h": -70.0, "solute_mg_l": 10.0},
        {"id": 2, "name": "e2", "isotope_d18o": -5.0, "isotope_d2h": -35.0, "solute_mg_l": 50.0},
    ]
    with pytest.raises(UnsolvableError) as excinfo:
        service.solve_mixture(sample, endmembers, 100, 1e-8)
    assert excinfo.value.reason == "no_quantitative_measurements"


def test_unsolvable_task_is_marked_without_producing_fractions(client):
    well = create_well(client, "W-R1")
    ids = create_endmembers(client)
    censored_only = add_sample(client, well["id"], "S-R1-CO", solute_mg_l=0.1, observations={"solute_mg_l": {"type": "censored"}})
    from app.database import get_connection

    connection = get_connection()
    # 直接构造一条历史遗留任务（绕过入队校验），验证执行路径同样拒绝输出比例
    connection.execute(
        "INSERT INTO hydro_inversions(sample_id,task_key,model_version,method,input_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
        (censored_only["id"], "legacy-task-key", "mix-obs", "weighted-least-squares", json.dumps({"endmember_ids": ids, "max_iterations": 100, "tolerance": 1e-8}), "2020-01-01T00:00:00+00:00", "2020-01-01T00:00:00+00:00"),
    )
    task_id = connection.execute("SELECT id FROM hydro_inversions WHERE task_key='legacy-task-key'").fetchone()[0]
    response = client.post(f"/api/hydro/inversions/{task_id}/run?worker_id=test")
    assert response.status_code == 422
    assert "unsolvable:no_quantitative_measurements" in response.json()["detail"]
    stored = connection.execute("SELECT status,error,result_json FROM hydro_inversions WHERE id=?", (task_id,)).fetchone()
    assert stored["status"] == "unsolvable"
    assert "no_quantitative_measurements" in stored["error"]
    assert stored["result_json"] == "{}"


def test_historical_sample_keeps_original_reported_values(client):
    well = create_well(client, "W-H1")
    ids = create_endmembers(client)
    from app.database import get_connection

    connection = get_connection()
    # 模拟迁移前的历史样本：按旧结构写入，没有 observation_types 信息
    connection.execute(
        "INSERT INTO hydro_samples(well_id,sample_code,sampled_at,isotope_d18o,isotope_d2h,solute_mg_l,detection_limit,measurement_error,quality_status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (well["id"], "S-H1-LEGACY", "2020-05-01T00:00:00+00:00", -7.5, None, 30.0, 0.1, 0.05, "usable", "2020-05-01T00:00:00+00:00"),
    )
    legacy = connection.execute("SELECT * FROM hydro_samples WHERE sample_code='S-H1-LEGACY'").fetchone()
    assert legacy["observation_types"] == "{}"
    result = run_inversion(client, legacy["id"], ids)
    # 历史样本按“有值=定量、空值=缺失”推断参与计算
    assert observation_of(result, "isotope_d18o")["type"] == "quantitative"
    assert observation_of(result, "isotope_d2h")["type"] == "missing"
    assert observation_of(result, "isotope_d2h")["participation"] == "excluded"
    assert observation_of(result, "solute_mg_l")["type"] == "quantitative"
    # 原始报告值保持原样，不被改写
    stored = connection.execute("SELECT * FROM hydro_samples WHERE id=?", (legacy["id"],)).fetchone()
    assert stored["isotope_d18o"] == -7.5
    assert stored["isotope_d2h"] is None
    assert stored["solute_mg_l"] == 30.0
    assert stored["observation_types"] == "{}"


def test_migration_adds_observation_types_without_touching_values(tmp_path, monkeypatch):
    db_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE hydro_samples (
         id INTEGER PRIMARY KEY AUTOINCREMENT, well_id INTEGER NOT NULL,
         sample_code TEXT NOT NULL UNIQUE, sampled_at TEXT NOT NULL, isotope_d18o REAL, isotope_d2h REAL,
         solute_mg_l REAL, detection_limit REAL NOT NULL, measurement_error REAL NOT NULL,
         quality_status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL
        );
        INSERT INTO hydro_samples(well_id,sample_code,sampled_at,isotope_d18o,isotope_d2h,solute_mg_l,detection_limit,measurement_error,quality_status,created_at)
        VALUES(1,'S-OLD','2020-01-01T00:00:00+00:00',-7.5,NULL,30.0,0.1,0.05,'usable','2020-01-01T00:00:00+00:00');
        """
    )
    connection.commit()
    connection.close()
    monkeypatch.setenv("TOWNSHIP_DATABASE_PATH", str(db_path))
    from app.database import close_connection

    close_connection()
    from app.hydro.service import ensure_schema

    ensure_schema()
    check = sqlite3.connect(db_path)
    check.row_factory = sqlite3.Row
    columns = {row[1] for row in check.execute("PRAGMA table_info(hydro_samples)")}
    assert "observation_types" in columns
    row = check.execute("SELECT * FROM hydro_samples WHERE sample_code='S-OLD'").fetchone()
    assert row["isotope_d18o"] == -7.5
    assert row["isotope_d2h"] is None
    assert row["solute_mg_l"] == 30.0
    assert row["observation_types"] == "{}"
    check.close()
    close_connection()


def test_result_lists_how_each_observation_participates(client):
    well = create_well(client, "W-P1")
    ids = create_endmembers(client)
    sample = add_sample(
        client,
        well["id"],
        "S-P1",
        isotope_d18o=-7.5,
        solute_mg_l=100.0,
        observations={"isotope_d18o": {"type": "quantitative"}, "isotope_d2h": {"type": "missing"}, "solute_mg_l": {"type": "censored"}},
    )
    assert sample["quality_status"] == "usable"
    sample_types = {obs["channel"]: obs["type"] for obs in sample["observations"]}
    assert sample_types == {"isotope_d18o": "quantitative", "isotope_d2h": "missing", "solute_mg_l": "censored"}
    result = run_inversion(client, sample["id"], ids)
    by_channel = {entry["channel"]: entry for entry in result["observations"]}
    assert set(by_channel) == {"isotope_d18o", "isotope_d2h", "solute_mg_l"}
    assert by_channel["isotope_d18o"]["participation"] == "squared-residual"
    assert by_channel["isotope_d18o"]["reported"] == -7.5
    assert by_channel["isotope_d2h"]["participation"] == "excluded"
    assert by_channel["isotope_d2h"]["reported"] is None
    assert by_channel["solute_mg_l"]["participation"] == "one-sided-inequality"
    assert by_channel["solute_mg_l"]["reported"] == 100.0
    assert by_channel["solute_mg_l"]["bound"] == 100.0


def test_observation_type_validation(client):
    well = create_well(client, "W-V1")
    base = {"sampled_at": "2026-09-24T08:00:00+00:00", "detection_limit": 0.1, "measurement_error": 0.05}

    # 左删失必须在数值栏填写报告给出的检出限
    response = client.post(f"/api/hydro/wells/{well['id']}/samples", json={**base, "sample_code": "S-V1-1", "observations": {"solute_mg_l": {"type": "censored"}}})
    assert response.status_code == 422
    assert "invalid_observation" in response.json()["detail"]

    # 缺失观测不应携带数值
    response = client.post(f"/api/hydro/wells/{well['id']}/samples", json={**base, "sample_code": "S-V1-2", "isotope_d2h": -50, "observations": {"isotope_d2h": {"type": "missing"}}})
    assert response.status_code == 422
    assert "invalid_observation" in response.json()["detail"]

    # 定量观测必须提供数值
    response = client.post(f"/api/hydro/wells/{well['id']}/samples", json={**base, "sample_code": "S-V1-3", "observations": {"isotope_d18o": {"type": "quantitative"}}})
    assert response.status_code == 422
    assert "invalid_observation" in response.json()["detail"]

    # 未知通道名被拒绝
    response = client.post(f"/api/hydro/wells/{well['id']}/samples", json={**base, "sample_code": "S-V1-4", "observations": {"unknown_channel": {"type": "censored"}}})
    assert response.status_code == 422


def test_quality_status_reflects_observation_types(client):
    well = create_well(client, "W-Q1")
    solvable = add_sample(client, well["id"], "S-Q1-1", isotope_d18o=-7.5, solute_mg_l=0.1, observations={"solute_mg_l": {"type": "censored"}})
    assert solvable["quality_status"] == "usable"
    # 两项均为左删失：旧规则（按非空计数）会误判为 usable，新规则判定信息不足
    censored_only = add_sample(client, well["id"], "S-Q1-2", isotope_d18o=-8.0, solute_mg_l=0.1, observations={"isotope_d18o": {"type": "censored"}, "solute_mg_l": {"type": "censored"}})
    assert censored_only["quality_status"] == "incomplete"
