"""Authoritative offline schema for the private Phase 1.5 fixture."""
from __future__ import annotations
import hashlib, json, math, re, unicodedata, wave
from pathlib import Path
from typing import Any

POLICY_VERSION="ru-nfkc-casefold-yo-punct-ws-v1"; SEMANTIC_SCHEMA_VERSION="jarvis-semantic-frame-v1"
INTENT_RULES={
 "open_application":({"application"},{"application":str}), "window_control":({"action","window","placement","layout","direction","monitor"},{"action":str,"window":str,"placement":str,"layout":str,"direction":str,"monitor":int}),
 "browser_control":({"action","query","url"},{"action":str,"query":str,"url":str}), "gesture_mode":({"action","enabled"},{"action":str,"enabled":bool}), "workspace_control":({"action","workspace"},{"action":str,"workspace":str}),
 "set_reminder":({"minutes","reminder_text","clock_time","day"},{"minutes":int,"reminder_text":str,"clock_time":str,"day":str}), "cancel_reminder":(set(),{}), "get_current_time":(set(),{}), "list_applications":(set(),{}), "list_reminders":(set(),{}), "cancel":(set(),{}), "confirm":(set(),{}), "decline":(set(),{}), "undo":(set(),{}), "general_chat":(set(),{}), "unknown":(set(),{}), "negated_command":(set(),{}), "unsupported_command":(set(),{}), "wake_greeting":(set(),{}),
 "file_control":({"action","query","path","new_name"},{"action":str,"query":str,"path":str,"new_name":str}), "system_control":({"action","setting"},{"action":str,"setting":str})}
REQUIRED={"schema_version","id","path","sha256","expected_speech","reference_text","semantic_scored","expected_actions","risk_class","acoustic_condition","recording_instructions","tags"}
def normalize(text:str)->str: return re.sub(r"\s+"," ",re.sub(r"[^\w\s]"," ",unicodedata.normalize("NFKC",text).casefold().replace("ё","е"),flags=re.UNICODE)).strip()
def sha256(path:Path)->str:
 h=hashlib.sha256();
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def plan_hash(root:Path)->str:return sha256(root/"phrase_plan.json")
def _rows(path:Path)->list[dict]: return json.loads(path.read_text(encoding="utf8"))
def validate_action(action:Any)->None:
 if not isinstance(action,dict) or set(action)!={"intent","slots"}:raise ValueError("action must contain exactly intent and slots")
 intent,slots=action["intent"],action["slots"]
 if intent not in INTENT_RULES:raise ValueError(f"unknown production intent: {intent}")
 if not isinstance(slots,dict):raise ValueError("action slots must be an object")
 allowed,types=INTENT_RULES[intent]
 if set(slots)-allowed:raise ValueError(f"unknown slots for {intent}: {sorted(set(slots)-allowed)}")
 for name,value in slots.items():
  if type(value) is not types[name]:raise ValueError(f"invalid type for {intent}.{name}")
def validate_plan(root:Path)->list[dict]:
 plan=_rows(root/"phrase_plan.json")
 if len(plan)!=70:raise ValueError(f"plan must contain exactly 70 cases, got {len(plan)}")
 ids=set(); noise=0
 for row in plan:
  if row.get("schema_version")!=SEMANTIC_SCHEMA_VERSION:raise ValueError("plan schema version mismatch")
  if not row.get("id") or row["id"] in ids:raise ValueError("duplicate/empty plan ID")
  ids.add(row["id"])
  if not isinstance(row.get("semantic_scored"),bool) or not isinstance(row.get("expected_actions"),list):raise ValueError("invalid semantic fields")
  for action in row["expected_actions"]:validate_action(action)
  if not row["semantic_scored"] and row["expected_actions"]:raise ValueError("unscored case has actions")
  cond,prompt=row["acoustic_condition"],str(row["prompt"]).casefold()
  if cond=="ordinary_room_noise" and row["expected_speech"]:noise+=1
  if "silence" in prompt and cond!="silence":raise ValueError("silence prompt has contradictory condition")
  if "keyboard" in prompt and cond!="keyboard_noise":raise ValueError("keyboard prompt has contradictory condition")
  if not row["expected_speech"] and cond in {"quiet_room","ordinary_room_noise"}:raise ValueError("non-speech case has speech condition")
 if noise!=20:raise ValueError(f"expected exactly 20 ordinary-room-noise speech cases, got {noise}")
 return plan
def _consent(root:Path)->dict:
 p=root/"fixture-consent.json"
 if not p.is_file():raise ValueError("missing dedicated fixture consent")
 c=json.loads(p.read_text(encoding="utf8"))
 if not c.get("local_processing_approved") or c.get("network_allowed") is not False or c.get("commit_allowed") is not False:raise ValueError("fixture consent privacy/no-action requirements are not approved")
 if c.get("fixture_root")!=str(root.resolve()) or c.get("plan_sha256")!=plan_hash(root):raise ValueError("fixture consent is not bound to this fixture root and plan hash")
 return c
def validate_manifest(root:Path,*,require_complete:bool=True)->list[dict]:
 plan=validate_plan(root); _consent(root); manifest=root/"manifest.jsonl"
 if not manifest.is_file():raise ValueError("missing manifest")
 rows=[json.loads(line) for line in manifest.read_text(encoding="utf8").splitlines() if line.strip()]; by_id={r.get("id"):r for r in rows}
 if len(by_id)!=len(rows):raise ValueError("duplicate manifest IDs")
 plan_by={r["id"]:r for r in plan}
 if require_complete and set(by_id)!=set(plan_by):raise ValueError("manifest IDs do not exactly match approved plan")
 for ident,row in by_id.items():
  if ident not in plan_by or set(row)<REQUIRED:raise ValueError("unknown/incomplete manifest row")
  expected=plan_by[ident]
  for key in REQUIRED-{"path","sha256"}:
   if row.get(key)!=expected.get(key):raise ValueError(f"manifest metadata differs from approved plan: {ident}.{key}")
  rel=Path(row["path"])
  if rel.is_absolute() or ".." in rel.parts or rel.parent!=Path("audio"):raise ValueError("unsafe fixture path")
  path=(root/rel).resolve()
  if root.resolve() not in path.parents or not path.is_file() or sha256(path)!=row["sha256"]:raise ValueError("missing or checksum-mismatched WAV")
  with wave.open(str(path),"rb") as w:
   if (w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getcomptype())!=(1,2,16000,"NONE"):raise ValueError("non PCM 16k mono WAV")
   if not .25<=w.getnframes()/16000<=12:raise ValueError("invalid WAV duration")
 return rows
