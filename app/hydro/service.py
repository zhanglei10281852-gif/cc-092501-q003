from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from typing import Any

from app.database import get_connection, transaction
from app.hydro.schemas import OBSERVATION_CHANNELS


SCHEMA = """
CREATE TABLE IF NOT EXISTS hydro_wells (
 id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
 latitude REAL NOT NULL, longitude REAL NOT NULL, aquifer TEXT NOT NULL, screen_depth_m REAL NOT NULL,
 status TEXT NOT NULL DEFAULT 'active', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hydro_endmembers (
 id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, isotope_d18o REAL NOT NULL,
 isotope_d2h REAL NOT NULL, solute_mg_l REAL NOT NULL, uncertainty REAL NOT NULL,
 version TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1)), created_at TEXT NOT NULL,
 UNIQUE(name,version)
);
CREATE TABLE IF NOT EXISTS hydro_samples (
 id INTEGER PRIMARY KEY AUTOINCREMENT, well_id INTEGER NOT NULL REFERENCES hydro_wells(id) ON DELETE RESTRICT,
 sample_code TEXT NOT NULL UNIQUE, sampled_at TEXT NOT NULL, isotope_d18o REAL, isotope_d2h REAL,
 solute_mg_l REAL, detection_limit REAL NOT NULL, measurement_error REAL NOT NULL,
 quality_status TEXT NOT NULL DEFAULT 'pending',
 observation_types TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hydro_inversions (
 id INTEGER PRIMARY KEY AUTOINCREMENT, sample_id INTEGER NOT NULL REFERENCES hydro_samples(id) ON DELETE RESTRICT,
 task_key TEXT NOT NULL UNIQUE, model_version TEXT NOT NULL, method TEXT NOT NULL,
 input_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued', attempts INTEGER NOT NULL DEFAULT 0,
 worker_id TEXT NOT NULL DEFAULT '', result_json TEXT NOT NULL DEFAULT '{}', error TEXT NOT NULL DEFAULT '',
 created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hydro_transport_runs (
 id INTEGER PRIMARY KEY AUTOINCREMENT, well_id INTEGER NOT NULL REFERENCES hydro_wells(id) ON DELETE RESTRICT,
 task_key TEXT NOT NULL UNIQUE, model_version TEXT NOT NULL, input_json TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'queued', result_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hydro_audit (
 id INTEGER PRIMARY KEY AUTOINCREMENT, resource_type TEXT NOT NULL, resource_id INTEGER,
 action TEXT NOT NULL, actor TEXT NOT NULL, payload_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hydro_samples_well ON hydro_samples(well_id,sampled_at);
CREATE INDEX IF NOT EXISTS idx_hydro_inversions_status ON hydro_inversions(status,created_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_schema() -> None:
    connection = get_connection()
    connection.executescript(SCHEMA)
    # 兼容历史库：仅新增 observation_types 列（默认 '{}'），不改动任何已存储的报告值。
    columns = {row[1] for row in connection.execute("PRAGMA table_info(hydro_samples)")}
    if "observation_types" not in columns:
        connection.execute("ALTER TABLE hydro_samples ADD COLUMN observation_types TEXT NOT NULL DEFAULT '{}'")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row else None


class UnsolvableError(ValueError):
    """可用观测信息不足，反演不可求解。reason 为机器可读原因码。"""

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        super().__init__(f"unsolvable:{reason}:{detail}")


SOLVABILITY_DETAILS = {
    "no_quantitative_measurements": "样本没有定量观测（全部缺测或仅有低于检出限的左删失观测），无法锚定混合比例",
    "insufficient_measurements": "有效观测不足：定量与左删失观测合计少于 2 项，无法约束混合比例",
}


def resolve_observations(sample: Any) -> list[dict[str, Any]]:
    """解析样本各通道的观测类型。

    新样本在写入时已显式记录类型；历史样本（observation_types 为 '{}'）按
    “有值=定量、空值=缺失”推断，存储的原始报告值保持不变。
    """
    data = dict(sample)
    raw = data.get("observation_types")
    declared = raw if isinstance(raw, dict) else (json.loads(raw) if raw else {})
    resolved = []
    for channel in OBSERVATION_CHANNELS:
        value = data.get(channel)
        obs_type = declared.get(channel)
        if obs_type is None:
            obs_type = "quantitative" if value is not None else "missing"
        resolved.append({"channel": channel, "type": obs_type, "value": value})
    return resolved


def solvability_problem(types: list[str]) -> str | None:
    """返回不可求解原因码；可求解时返回 None。

    规则：至少 1 项定量观测（删失观测只是不等式约束，无法单独锚定比例），
    且定量与左删失观测合计至少 2 项。
    """
    quantitative = sum(1 for obs_type in types if obs_type == "quantitative")
    usable = sum(1 for obs_type in types if obs_type != "missing")
    if quantitative == 0:
        return "no_quantitative_measurements"
    if usable < 2:
        return "insufficient_measurements"
    return None


def assert_solvable(types: list[str]) -> None:
    problem = solvability_problem(types)
    if problem is not None:
        raise UnsolvableError(problem, SOLVABILITY_DETAILS[problem])


class HydroService:
    def __init__(self, connection: sqlite3.Connection | None = None):
        self.connection = connection or get_connection()
        ensure_schema()

    def create_well(self, payload: dict[str, Any], actor: str = "researcher") -> dict[str, Any]:
        now = _now()
        with transaction(immediate=True) as connection:
            cursor = connection.execute("INSERT INTO hydro_wells(code,name,latitude,longitude,aquifer,screen_depth_m,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)", (payload["code"],payload["name"],payload["latitude"],payload["longitude"],payload["aquifer"],payload["screen_depth_m"],now,now))
            well_id = cursor.lastrowid
            connection.execute("INSERT INTO hydro_audit(resource_type,resource_id,action,actor,payload_json,created_at) VALUES('well',?,?,?,?,?)", (well_id,"create",actor,json.dumps(payload,ensure_ascii=False),now))
            return dict(connection.execute("SELECT * FROM hydro_wells WHERE id=?",(well_id,)).fetchone())

    def get_well(self, well_id: int) -> dict[str, Any] | None:
        well = self.connection.execute("SELECT * FROM hydro_wells WHERE id=?",(well_id,)).fetchone()
        if well is None: return None
        result = dict(well)
        samples = [dict(r) for r in self.connection.execute("SELECT * FROM hydro_samples WHERE well_id=? ORDER BY sampled_at,id",(well_id,)).fetchall()]
        for sample in samples:
            sample["observations"] = resolve_observations(sample)
        result["samples"] = samples
        return result

    def delete_well(self, well_id: int) -> bool:
        with transaction(immediate=True) as connection:
            cursor = connection.execute("DELETE FROM hydro_wells WHERE id=?",(well_id,))
            if cursor.rowcount == 0: raise KeyError("well_not_found")
            return True

    def create_endmember(self, payload: dict[str, Any]) -> dict[str, Any]:
        now=_now()
        with transaction(immediate=True) as connection:
            cursor=connection.execute("INSERT INTO hydro_endmembers(name,isotope_d18o,isotope_d2h,solute_mg_l,uncertainty,version,created_at) VALUES(?,?,?,?,?,?,?)",(payload["name"],payload["isotope_d18o"],payload["isotope_d2h"],payload["solute_mg_l"],payload["uncertainty"],payload["version"],now))
            return dict(connection.execute("SELECT * FROM hydro_endmembers WHERE id=?",(cursor.lastrowid,)).fetchone())

    def _normalize_observation_types(self, payload: dict[str, Any]) -> dict[str, str]:
        """校验并补全各通道观测类型；未声明的通道按“有值=定量、空值=缺失”推断。"""
        specs = payload.get("observations") or {}
        types: dict[str, str] = {}
        for channel in OBSERVATION_CHANNELS:
            value = payload.get(channel)
            spec = specs.get(channel)
            if spec is not None:
                obs_type = spec["type"] if isinstance(spec, dict) else spec.type
            else:
                obs_type = "quantitative" if value is not None else "missing"
            if obs_type == "quantitative" and value is None:
                raise ValueError(f"invalid_observation:{channel}:定量观测必须提供数值")
            if obs_type == "censored" and value is None:
                raise ValueError(f"invalid_observation:{channel}:左删失观测必须在数值栏填写报告给出的检出限")
            if obs_type == "missing" and value is not None:
                raise ValueError(f"invalid_observation:{channel}:缺失观测不应携带数值")
            types[channel] = obs_type
        return types

    def add_sample(self, well_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        if self.connection.execute("SELECT id FROM hydro_wells WHERE id=?",(well_id,)).fetchone() is None: raise KeyError("well_not_found")
        types=self._normalize_observation_types(payload)
        quality="usable" if solvability_problem(list(types.values())) is None else "incomplete"
        now=_now()
        with transaction(immediate=True) as connection:
            cursor=connection.execute("INSERT INTO hydro_samples(well_id,sample_code,sampled_at,isotope_d18o,isotope_d2h,solute_mg_l,detection_limit,measurement_error,quality_status,observation_types,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",(well_id,payload["sample_code"],payload["sampled_at"],payload.get("isotope_d18o"),payload.get("isotope_d2h"),payload.get("solute_mg_l"),payload["detection_limit"],payload["measurement_error"],quality,json.dumps(types,ensure_ascii=False),now))
            sample=dict(connection.execute("SELECT * FROM hydro_samples WHERE id=?",(cursor.lastrowid,)).fetchone())
            sample["observations"]=resolve_observations(sample)
            return sample

    def _project_simplex(self, values: list[float]) -> list[float]:
        clipped=[max(0.0,v) for v in values]
        total=sum(clipped)
        return [1/len(values)]*len(values) if total<=1e-15 else [v/total for v in clipped]

    def solve_mixture(self, sample: Any, endmembers: list[Any], max_iterations: int, tolerance: float) -> dict[str, Any]:
        """反演端元混合比例。

        观测按类型参与目标函数：
        - quantitative（定量值）：平方残差 ((预测-观测)/尺度)^2；
        - censored（左删失，低于检出限）：单侧残差 max(0,(预测-检出限)/尺度)^2，
          预测不高于检出限时无惩罚，检出限绝不被当作 0 或精确值；
        - missing（真正缺失）：不参与计算。
        信息不足时抛出 UnsolvableError 并给出机器可读原因，而不是输出看似精确的比例。
        """
        observations=resolve_observations(sample)
        types=[item["type"] for item in observations]
        reported=[item["value"] for item in observations]
        assert_solvable(types)
        quantitative=[k for k,obs_type in enumerate(types) if obs_type=="quantitative"]
        solute=reported[2]
        scale=[20.0,100.0,max(1.0,float(solute)) if solute is not None else 1.0]
        vectors=[[e["isotope_d18o"],e["isotope_d2h"],e["solute_mg_l"]] for e in endmembers]

        def predict(fractions: list[float]) -> list[float]:
            return [sum(fractions[j]*vector[k] for j,vector in enumerate(vectors)) for k in range(3)]

        def residuals(predicted: list[float]) -> list[float]:
            values=[]
            for k in range(3):
                if types[k]=="quantitative":
                    values.append((predicted[k]-float(reported[k]))/scale[k])
                elif types[k]=="censored":
                    values.append(max(0.0,(predicted[k]-float(reported[k]))/scale[k]))
                else:
                    values.append(0.0)
            return values

        fractions=[1/len(endmembers)]*len(endmembers)
        rate=0.08
        last=float("inf")
        converged=False
        iterations=0
        for iteration in range(1,max_iterations+1):
            iterations=iteration
            residual=residuals(predict(fractions))
            objective=sum(r*r for r in residual)+((sum(fractions)-1.0)*10)**2
            if abs(last-objective)<tolerance:
                converged=True
                break
            last=objective
            gradient=[2*sum(residual[k]*vector[k]/scale[k] for k in range(3)) for vector in vectors]
            fractions=self._project_simplex([f-rate*g for f,g in zip(fractions,gradient)])
        predicted=predict(fractions)
        residual=residuals(predicted)
        objective=sum(r*r for r in residual)+((sum(fractions)-1.0)*10)**2
        rmse=math.sqrt(sum(residual[k]**2 for k in quantitative)/len(quantitative))
        participation=[]
        for k,item in enumerate(observations):
            entry: dict[str, Any]={"channel":item["channel"],"type":item["type"],"reported":item["value"]}
            if item["type"]=="quantitative":
                entry.update(participation="squared-residual",predicted=predicted[k],scale=scale[k],residual=residual[k],contribution=residual[k]**2)
            elif item["type"]=="censored":
                bound=float(item["value"])
                entry.update(participation="one-sided-inequality",bound=bound,predicted=predicted[k],scale=scale[k],satisfied=predicted[k]<=bound+1e-9*max(1.0,abs(bound)),residual=residual[k],contribution=residual[k]**2)
            else:
                entry["participation"]="excluded"
            participation.append(entry)
        return {"fractions":[{"endmember_id":e["id"],"name":e["name"],"fraction":round(f,8)} for e,f in zip(endmembers,fractions)],"mass_balance":round(sum(fractions),10),"predicted":predicted,"rmse":rmse,"objective":objective,"observations":participation,"iterations":iterations,"converged":converged}

    def enqueue_inversion(self, sample_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        sample=self.connection.execute("SELECT * FROM hydro_samples WHERE id=?",(sample_id,)).fetchone()
        if sample is None: raise KeyError("sample_not_found")
        assert_solvable([item["type"] for item in resolve_observations(sample)])
        ids=sorted(set(payload["endmember_ids"]))
        endmembers=self.connection.execute(f"SELECT * FROM hydro_endmembers WHERE active=1 AND id IN ({','.join('?' for _ in ids)}) ORDER BY id",ids).fetchall()
        if len(endmembers)!=len(ids): raise ValueError("endmember_not_found")
        input_data={**payload,"endmember_ids":ids,"sample":dict(sample),"endmembers":[dict(e) for e in endmembers]}
        key=_digest(input_data); now=_now()
        with transaction(immediate=True) as connection:
            old=connection.execute("SELECT * FROM hydro_inversions WHERE task_key=?",(key,)).fetchone()
            if old: return dict(old)
            cursor=connection.execute("INSERT INTO hydro_inversions(sample_id,task_key,model_version,method,input_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",(sample_id,key,payload["model_version"],payload["method"],json.dumps(input_data,ensure_ascii=False),now,now))
            return dict(connection.execute("SELECT * FROM hydro_inversions WHERE id=?",(cursor.lastrowid,)).fetchone())

    def run_inversion(self, task_id: int, worker_id: str) -> dict[str, Any]:
        with transaction(immediate=True) as connection:
            task=connection.execute("SELECT * FROM hydro_inversions WHERE id=?",(task_id,)).fetchone()
            if task is None: raise KeyError("task_not_found")
            if task["status"]=="done": return dict(task)
            connection.execute("UPDATE hydro_inversions SET status='running',attempts=attempts+1,worker_id=?,updated_at=? WHERE id=?",(worker_id,_now(),task_id))
        data=json.loads(task["input_json"])
        sample=self.connection.execute("SELECT * FROM hydro_samples WHERE id=?",(task["sample_id"],)).fetchone()
        ids=data["endmember_ids"]
        endmembers=self.connection.execute(f"SELECT * FROM hydro_endmembers WHERE id IN ({','.join('?' for _ in ids)}) ORDER BY id",ids).fetchall()
        try: result=self.solve_mixture(sample,endmembers,data["max_iterations"],data["tolerance"])
        except UnsolvableError as exc:
            with transaction(immediate=True) as connection: connection.execute("UPDATE hydro_inversions SET status='unsolvable',error=?,updated_at=? WHERE id=?",(str(exc),_now(),task_id))
            raise
        except Exception as exc:
            with transaction(immediate=True) as connection: connection.execute("UPDATE hydro_inversions SET status='failed',error=?,updated_at=? WHERE id=?",(str(exc),_now(),task_id))
            raise
        with transaction(immediate=True) as connection:
            connection.execute("UPDATE hydro_inversions SET status='done',result_json=?,error='',updated_at=? WHERE id=?",(json.dumps(result,ensure_ascii=False),_now(),task_id))
            return dict(connection.execute("SELECT * FROM hydro_inversions WHERE id=?",(task_id,)).fetchone())

    def run_transport(self, well_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        if self.connection.execute("SELECT id FROM hydro_wells WHERE id=?",(well_id,)).fetchone() is None: raise KeyError("well_not_found")
        key=_digest({"well_id":well_id,**payload}); now=_now()
        old=self.connection.execute("SELECT * FROM hydro_transport_runs WHERE task_key=?",(key,)).fetchone()
        if old: return dict(old)
        points=[]; t=payload["step_days"]
        while t<=payload["duration_days"]+1e-12:
            d=payload["dispersion_m2_day"]; x=payload["distance_m"]; v=payload["velocity_m_day"]
            c=payload["source_concentration"]*math.exp(-((x-v*t)**2)/(4*d*t))*math.exp(-payload["decay_per_day"]*t)/math.sqrt(4*math.pi*d*t)
            points.append({"time_days":round(t,8),"concentration":c}); t+=payload["step_days"]
        peak=max(points,key=lambda p:p["concentration"])
        result={"points":points,"peak":peak,"arrival_time_days":payload["distance_m"]/payload["velocity_m_day"],"model_version":payload["model_version"]}
        with transaction(immediate=True) as connection:
            cursor=connection.execute("INSERT INTO hydro_transport_runs(well_id,task_key,model_version,input_json,status,result_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",(well_id,key,payload["model_version"],json.dumps(payload,ensure_ascii=False),"done",json.dumps(result,ensure_ascii=False),now,now))
            return dict(connection.execute("SELECT * FROM hydro_transport_runs WHERE id=?",(cursor.lastrowid,)).fetchone())
