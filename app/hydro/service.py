from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from typing import Any

from app.database import get_connection, transaction


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
 quality_status TEXT NOT NULL DEFAULT 'pending', observations_json TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
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
    columns = {row[1] for row in connection.execute("PRAGMA table_info(hydro_samples)")}
    if "observations_json" not in columns:
        connection.execute("ALTER TABLE hydro_samples ADD COLUMN observations_json TEXT NOT NULL DEFAULT ''")


TRACERS = ("isotope_d18o", "isotope_d2h", "solute_mg_l")
OBSERVATION_ROLES = {"quantitative": "residual", "censored": "censored_likelihood", "missing": "excluded"}


def _log_norm_cdf(z: float) -> float:
    """标准正态分布的对数累积概率 log Φ(z)。

    z 非常负时 erfc 会下溢为零，改用渐近展开 log Φ(z) ≈ -z²/2 - log(-z) - ½log(2π)，
    保证删失似然在预测值远高于检出限时仍然有限、可复现。
    """
    if z < -36.0:
        return -0.5 * z * z - math.log(-z) - 0.5 * math.log(2.0 * math.pi)
    return math.log(0.5 * math.erfc(-z / math.sqrt(2.0)))


def _mills_ratio(z: float) -> float:
    """米尔斯比 φ(z)/Φ(z)，即删失对数似然对预测值的梯度系数。"""
    if z < -36.0:
        return -z - 1.0 / z
    return math.exp(-0.5 * z * z - 0.5 * math.log(2.0 * math.pi) - _log_norm_cdf(z))


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row else None


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
        result["samples"] = [dict(r) for r in self.connection.execute("SELECT * FROM hydro_samples WHERE well_id=? ORDER BY sampled_at,id",(well_id,)).fetchall()]
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

    def _normalize_observations(self, payload: dict[str, Any]) -> dict[str, Any]:
        """把请求中的观测类型声明归一化为完整的三示踪剂规格并校验一致性。

        未声明的示踪剂按历史规则派生（空值→missing，非空→quantitative）；
        原始报告值不做任何改写，删失值只记录检出限阈值。
        """
        declared = payload.get("observations") or {}
        normalized: dict[str, Any] = {}
        for tracer in TRACERS:
            value = payload.get(tracer)
            spec = declared.get(tracer) or {}
            obs_type = getattr(spec.get("type"), "value", spec.get("type"))
            if obs_type is None:
                obs_type = "missing" if value is None else "quantitative"
            if obs_type not in OBSERVATION_ROLES:
                raise ValueError(f"unknown_observation_type:{obs_type}")
            if obs_type == "quantitative" and value is None:
                raise ValueError(f"quantitative_requires_value:{tracer}")
            if obs_type == "missing" and value is not None:
                raise ValueError(f"missing_forbids_value:{tracer}")
            entry: dict[str, Any] = {"type": obs_type}
            if obs_type == "censored":
                detection_limit = spec.get("detection_limit")
                if detection_limit is None:
                    detection_limit = payload.get("detection_limit") or 0.0
                if detection_limit <= 0:
                    raise ValueError(f"censored_requires_detection_limit:{tracer}")
                entry["detection_limit"] = float(detection_limit)
            normalized[tracer] = entry
        return normalized

    def add_sample(self, well_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        if self.connection.execute("SELECT id FROM hydro_wells WHERE id=?",(well_id,)).fetchone() is None: raise KeyError("well_not_found")
        observations=self._normalize_observations(payload)
        measured=sum(1 for o in observations.values() if o["type"] in ("quantitative","censored"))
        quality="usable" if measured>=2 else "incomplete"
        now=_now()
        with transaction(immediate=True) as connection:
            cursor=connection.execute("INSERT INTO hydro_samples(well_id,sample_code,sampled_at,isotope_d18o,isotope_d2h,solute_mg_l,detection_limit,measurement_error,quality_status,observations_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",(well_id,payload["sample_code"],payload["sampled_at"],payload.get("isotope_d18o"),payload.get("isotope_d2h"),payload.get("solute_mg_l"),payload["detection_limit"],payload["measurement_error"],quality,json.dumps(observations,ensure_ascii=False,sort_keys=True),now))
            return dict(connection.execute("SELECT * FROM hydro_samples WHERE id=?",(cursor.lastrowid,)).fetchone())

    def _project_simplex(self, values: list[float]) -> list[float]:
        """把点精确投影到概率单纯形上（Duchi 等人的欧氏投影算法）。"""
        ordered=sorted(values,reverse=True)
        cumulative=0.0
        theta=0.0
        for i,v in enumerate(ordered):
            cumulative+=v
            candidate=(cumulative-1.0)/(i+1)
            if v-candidate>0: theta=candidate
        return [max(v-theta,0.0) for v in values]

    def _resolve_observations(self, sample: Any) -> list[dict[str, Any]]:
        """读取样本的观测类型规格；历史行（observations_json 为空）按空值规则派生，报告值保持原样。"""
        raw = sample["observations_json"] if "observations_json" in sample.keys() else ""
        stored = json.loads(raw) if raw else {}
        observations=[]
        for tracer in TRACERS:
            value=sample[tracer]
            spec=stored.get(tracer) or {}
            obs_type=spec.get("type") or ("missing" if value is None else "quantitative")
            detection_limit=spec.get("detection_limit")
            if obs_type=="censored":
                if detection_limit is None: detection_limit=float(sample["detection_limit"])
                detection_limit=float(detection_limit)
                if detection_limit<=0: raise ValueError(f"censored_requires_detection_limit:{tracer}")
            observations.append({"tracer":tracer,"type":obs_type,"value":value,"detection_limit":detection_limit})
        return observations

    def _unsolvable(self, reason: str, detail: str, observations: list[dict[str, Any]], summary: dict[str, int]) -> dict[str, Any]:
        accounting=[]
        for o in observations:
            entry={"tracer":o["tracer"],"type":o["type"],"role":OBSERVATION_ROLES[o["type"]]}
            if o["type"]=="quantitative": entry["value"]=o["value"]
            if o["type"]=="censored": entry.update(value=o["value"],detection_limit=o["detection_limit"])
            accounting.append(entry)
        return {"status":"unsolvable","reason":reason,"reason_detail":detail,"fractions":None,**summary,"observations":accounting}

    def solve_mixture(self, sample: sqlite3.Row, endmembers: list[sqlite3.Row], max_iterations: int, tolerance: float) -> dict[str, Any]:
        observations=self._resolve_observations(sample)
        measurement_error=float(sample["measurement_error"])
        scales=[]
        for k,o in enumerate(observations):
            if k==0: scales.append(20.0)
            elif k==1: scales.append(100.0)
            elif o["type"]=="censored": scales.append(max(1.0,float(o["detection_limit"])))
            else: scales.append(max(1.0,float(sample["solute_mg_l"] or 1)))
        vectors=[[e["isotope_d18o"],e["isotope_d2h"],e["solute_mg_l"]] for e in endmembers]
        required=len(endmembers)-1
        quantitative=[k for k,o in enumerate(observations) if o["type"]=="quantitative"]
        censored=[k for k,o in enumerate(observations) if o["type"]=="censored"]
        missing=[k for k,o in enumerate(observations) if o["type"]=="missing"]
        informative=[k for k in quantitative if max(v[k] for v in vectors)-min(v[k] for v in vectors)>1e-12]
        summary={"quantitative_tracers":len(quantitative),"informative_quantitative_tracers":len(informative),"required_quantitative_tracers":required,"censored_tracers":len(censored),"missing_tracers":len(missing)}
        if not quantitative:
            return self._unsolvable("no_quantitative_measurements","样本没有定量测量值：低于检出限的删失数据与缺测项只能提供不等式约束，无法唯一确定端元比例",observations,summary)
        if len(informative)<required:
            return self._unsolvable("underdetermined_system",f"有效定量示踪剂只有{len(informative)}个，少于{len(endmembers)}个端元混合所需的最少{required}个，方程组欠定，无法唯一确定端元比例",observations,summary)

        def evaluate(fractions: list[float]) -> tuple[float, list[float]]:
            predicted=[sum(fractions[j]*vectors[j][k] for j in range(len(vectors))) for k in range(3)]
            value=((sum(fractions)-1.0)*10)**2
            for k in quantitative:
                residual=(predicted[k]-float(observations[k]["value"]))/scales[k]
                value+=residual*residual
            for k in censored:
                sigma=max(measurement_error*scales[k],1e-12)
                value+=2.0*measurement_error**2*(-_log_norm_cdf((observations[k]["detection_limit"]-predicted[k])/sigma))
            return value,predicted

        fractions=[1/len(endmembers)]*len(endmembers)
        rate=0.08
        last=float("inf")
        objective=float("inf")
        predicted=[sum(fractions[j]*vectors[j][k] for j in range(len(vectors))) for k in range(3)]
        converged=False
        iteration=-1
        for iteration in range(max_iterations):
            objective,predicted=evaluate(fractions)
            if abs(last-objective)<tolerance:
                converged=True
                break
            last=objective
            gradient=[]
            for j in range(len(vectors)):
                g=0.0
                for k in quantitative:
                    g+=2.0*((predicted[k]-float(observations[k]["value"]))/scales[k])*vectors[j][k]/scales[k]
                for k in censored:
                    sigma=max(measurement_error*scales[k],1e-12)
                    g+=2.0*measurement_error**2*_mills_ratio((observations[k]["detection_limit"]-predicted[k])/sigma)/sigma*vectors[j][k]
                gradient.append(g)
            step=rate
            for _ in range(30):
                candidate=self._project_simplex([f-step*g for f,g in zip(fractions,gradient)])
                candidate_objective,_=evaluate(candidate)
                if candidate_objective<objective-1e-15:
                    fractions=candidate
                    break
                step*=0.5
            else:
                converged=True
                break
        objective,predicted=evaluate(fractions)
        result_observations=[]
        censored_log_likelihood=0.0
        for k,o in enumerate(observations):
            entry={"tracer":o["tracer"],"type":o["type"],"role":OBSERVATION_ROLES[o["type"]]}
            if o["type"]=="quantitative":
                residual=(predicted[k]-float(o["value"]))/scales[k]
                entry.update(value=o["value"],scale=scales[k],weight=round(1.0/scales[k],8),predicted=predicted[k],residual=residual)
            elif o["type"]=="censored":
                sigma=max(measurement_error*scales[k],1e-12)
                z=(o["detection_limit"]-predicted[k])/sigma
                log_likelihood=_log_norm_cdf(z)
                censored_log_likelihood+=log_likelihood
                entry.update(value=o["value"],detection_limit=o["detection_limit"],predicted=predicted[k],z_score=z,log_likelihood=log_likelihood)
            result_observations.append(entry)
        residuals=[(predicted[k]-float(observations[k]["value"]))/scales[k] for k in quantitative]
        rmse=math.sqrt(sum(r*r for r in residuals)/len(residuals))
        return {"status":"solved","reason":None,"reason_detail":None,"fractions":[{"endmember_id":e["id"],"name":e["name"],"fraction":round(f,8)} for e,f in zip(endmembers,fractions)],"mass_balance":round(sum(fractions),10),"predicted":predicted,"rmse":rmse,"objective":objective,"censored_log_likelihood":censored_log_likelihood,"iterations":iteration+1,"converged":converged,**summary,"observations":result_observations}

    def enqueue_inversion(self, sample_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        sample=self.connection.execute("SELECT * FROM hydro_samples WHERE id=?",(sample_id,)).fetchone()
        if sample is None: raise KeyError("sample_not_found")
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
