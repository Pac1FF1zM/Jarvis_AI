"""Interactive, local-only fixture recorder. It imports no Jarvis runtime."""
from __future__ import annotations
import argparse, json, os, queue, shutil, sys, tempfile, time, wave
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from experiments.parakeet.fixtures import SEMANTIC_SCHEMA_VERSION,plan_hash,sha256,validate_manifest,validate_plan
def A(intent,**slots):return {"intent":intent,"slots":slots}
def _row(ident,text,actions,tag,*,speech=True,condition="quiet_room",prompt=None,scored=True):return {"schema_version":SEMANTIC_SCHEMA_VERSION,"id":ident,"prompt":prompt or f"Say: {text}","expected_speech":speech,"reference_text":text,"semantic_scored":scored,"expected_actions":actions,"risk_class":"negative" if tag=="negative" else "actionable" if actions else "non_action","acoustic_condition":condition,"recording_instructions":f"Record only in {condition}; do not change the condition.","tags":[tag]}
def plan():
 apps=[("калькулятор","calculator"),("блокнот","notepad"),("браузер","browser"),("проводник","explorer"),("пейнт","paint"),("дискорд","discord")]
 rows=[_row(f"command-{i:02d}",f"открой {name}",[A("open_application",application=key)],"command") for i,(name,key) in enumerate(apps,1)]
 rows += [_row("command-07","закрой калькулятор",[A("window_control",action="close",window="калькулятор")],"command"),_row("command-08","найди погоду в ташкенте",[A("browser_control",action="search",query="погоду в ташкенте")],"command"),_row("command-09","включи жестовый режим",[A("gesture_mode",action="enable",enabled=True)],"command"),_row("command-10","выключи жестовый режим",[A("gesture_mode",action="disable",enabled=False)],"command"),_row("command-11","сохрани рабочее пространство как работа",[A("workspace_control",action="capture",workspace="работа")],"command"),_row("command-12","включи игровой режим",[A("workspace_control",action="launch",workspace="gaming")],"command"),_row("command-13","напомни через пять минут купить хлеб",[A("set_reminder",minutes=5,reminder_text="купить хлеб")],"command"),_row("command-14","отмени напоминание",[A("cancel_reminder")],"command"),_row("command-15","какое сейчас время",[A("get_current_time")],"command"),_row("command-16","остановись",[A("cancel")],"command"),_row("command-17","отмена",[A("cancel")],"command"),_row("command-18","открой диспетчер задач",[A("open_application",application="task_manager")],"command"),_row("command-19","закрой браузер",[A("window_control",action="close",window="браузер")],"command"),_row("command-20","запусти калькулятор и блокнот",[A("open_application",application="calculator"),A("open_application",application="notepad")],"command"),_row("command-21","покажи открытые окна",[A("window_control",action="list")],"command"),_row("command-22","создай новую вкладку",[A("browser_control",action="new_tab")],"command"),_row("command-23","покажи приложения",[A("list_applications")],"command"),_row("command-24","покажи напоминания",[A("list_reminders")],"command"),_row("command-25","сверни все окна",[A("window_control",action="minimize_all")],"command"),_row("command-26","покажи рабочий стол",[A("window_control",action="show_desktop")],"command"),_row("command-27","переключись на браузер",[A("window_control",action="switch",window="браузер")],"command"),_row("command-28","найди файл отчет",[A("file_control",action="find",query="отчет")],"command"),_row("command-29","открой настройки звука",[A("system_control",action="open_settings",setting="sound")],"command"),_row("command-30","отмени последнее действие",[A("undo")],"command")]
 code=[("open calculator",[A("open_application",application="calculator")]),("запусти Discord",[A("open_application",application="discord")]),("открой Paint",[A("open_application",application="paint")]),("найди OpenAI",[A("browser_control",action="search",query="openai")]),("открой browser",[A("open_application",application="browser")]),("открой Notepad",[A("open_application",application="notepad")]),("найди ChatGPT",[A("browser_control",action="search",query="chatgpt")]),("открой File Explorer",[A("open_application",application="explorer")]),("открой Task Manager",[A("open_application",application="task_manager")]),("позвони Ивану",[])]
 rows += [_row(f"code-switch-{i:02d}",t,a,"code_switch",scored=bool(a)) for i,(t,a) in enumerate(code,1)]
 for i,t in enumerate(["эм, открой пожалуйста калькулятор","открой браузер, нет, лучше блокнот","можешь напомнить через пятнадцать минут","я хотел бы узнать который сейчас час","сначала открой проводник потом калькулятор","пожалуйста остановись","нет отмена","может быть запусти дискорд","открой телеграм когда будет удобно","найди рецепт плова в интернете"],1):rows.append(_row(f"natural-{i:02d}",t,[],"natural",scored=True))
 for i,t in enumerate(["расскажи интересный факт","как дела","доброе утро","что ты умеешь","поболтаем о музыке","я просто проверяю микрофон","какая красивая погода","не знаю что сказать","привет джарвис","расскажи шутку"],1):rows.append(_row(f"unknown-{i:02d}",t,[],"unknown",scored=True))
 neg=[("silence","Record 2 seconds of silence",False,"", "silence"),("keyboard","Record keyboard-only noise",False,"","keyboard_noise"),("room","Record room noise only",False,"","room_noise"),("wake","Say: джарвис",True,"джарвис","quiet_room"),("incomplete","Say: открой",True,"открой","quiet_room"),("noise-word","Say: случайный шум",True,"случайный шум","quiet_room"),("hello","Say: алло",True,"алло","quiet_room"),("filler","Say: эээ",True,"эээ","quiet_room"),("negated","Say: не выполняй это",True,"не выполняй это","quiet_room"),("background","Say: фоновый разговор",True,"фоновый разговор","quiet_room")]
 rows += [_row(f"negative-{i:02d}",text,[],"negative",speech=speech,condition=condition,prompt=prompt,scored=True) for i,(tag,prompt,speech,text,condition) in enumerate(neg,1)]
 # Explicit assignment, never inferred from numeric portions of IDs.
 for row in rows[:20]:row["acoustic_condition"]="ordinary_room_noise";row["recording_instructions"]="Record spoken phrase under ordinary room noise; do not use quiet-room audio."
 return rows
def root(a):return Path(a.root).resolve()
def _empty(r):return not any((r/"audio").glob("*.wav")) and not (r/"manifest.jsonl").read_text(encoding="utf8").strip() if (r/"manifest.jsonl").exists() else True
def init(r):
 if not _empty(r):raise SystemExit("refusing to modify a non-empty fixture")
 (r/"audio").mkdir(parents=True,exist_ok=True);(r/"results").mkdir(exist_ok=True);(r/"phrase_plan.json").write_text(json.dumps(plan(),ensure_ascii=False,indent=2),encoding="utf8");(r/"manifest.jsonl").touch();print("plan regenerated; consent must be re-approved for its new hash")
def devices():
 import sounddevice as sd
 for i,d in enumerate(sd.query_devices()):print(f"{i}: {d['name']} input_channels={d['max_input_channels']} default_rate={d['default_samplerate']}")
def test_device(device):
 import sounddevice as sd
 import numpy as np
 input("Press Enter to test the selected microphone for 2 seconds: "); q=queue.Queue();status=[]
 def cb(indata,frames,clock,state):q.put(indata.copy());status.append(str(state)) if state else None
 with sd.InputStream(samplerate=16000,channels=1,dtype="int16",device=device,callback=cb):time.sleep(2)
 data=np.concatenate(list(q.queue)) if not q.empty() else np.empty((0,1),dtype=np.int16);peak=int(abs(data).max()) if len(data) else 0;rms=float(np.sqrt(np.mean(data.astype(float)**2))) if len(data) else 0.;clips=int((abs(data)>=32760).sum());print(f"device={device} samples={len(data)} peak={peak} rms={rms:.1f} clips={clips} callback_status={status}")
def _entry(row,path):return {k:row[k] for k in ("schema_version","id","expected_speech","reference_text","semantic_scored","expected_actions","risk_class","acoustic_condition","recording_instructions","tags")}|{"path":f"audio/{row['id']}.wav","sha256":sha256(path)}
def record(r,ident,device,*,replace=False):
 consent=json.loads((r/"fixture-consent.json").read_text(encoding="utf8"))
 if not consent.get("local_processing_approved") or consent.get("plan_sha256")!=plan_hash(r):raise SystemExit("recording blocked: fixture consent is not approved for this exact plan")
 row=next((x for x in json.loads((r/"phrase_plan.json").read_text(encoding="utf8")) if x["id"]==ident),None)
 if not row:raise SystemExit("unknown fixture ID")
 print(f"PHRASE: {row['prompt']}\nCONDITION: {row['acoustic_condition']}\nACTIONS: {row['expected_actions']}");input("Press Enter to START microphone (Ctrl+C cancels): ")
 import sounddevice as sd, numpy as np
 q=queue.Queue();statuses=[];tmp=r/"audio"/f".{ident}.tmp.wav";print("MICROPHONE ACTIVE — press Enter to stop.")
 def cb(indata,frames,time,status):q.put(indata.copy());statuses.append(str(status)) if status else None
 try:
  with sd.InputStream(samplerate=16000,channels=1,dtype="int16",device=device,callback=cb):input()
 except KeyboardInterrupt:print("cancelled");return False
 data=np.concatenate(list(q.queue)) if not q.empty() else np.empty((0,1),dtype=np.int16)
 with wave.open(str(tmp),"wb") as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(data.tobytes())
 peak=int(abs(data).max()) if len(data) else 0;rms=float(np.sqrt(np.mean(data.astype(float)**2))) if len(data) else 0.;clips=int((abs(data)>=32760).sum());duration=len(data)/16000;print(f"duration={duration:.2f}s peak={peak} rms={rms:.1f} clips={clips} callback_errors={statuses}")
 if not .25<=duration<=12 or clips>10 or (row["expected_speech"] and rms<100):tmp.unlink(missing_ok=True);print("rejected recording");return False
 while True:
  answer=input("[p]lay [s]ave [r]etry [k]skip [q]uit: ").casefold().strip()
  if answer=="p":import winsound;winsound.PlaySound(str(tmp),winsound.SND_FILENAME)
  elif answer=="s":
   final=r/"audio"/f"{ident}.wav";os.replace(tmp,final);rows=[json.loads(x) for x in (r/"manifest.jsonl").read_text(encoding="utf8").splitlines() if x.strip()];rows=[x for x in rows if x["id"]!=ident];rows.append(_entry(row,final));fd,name=tempfile.mkstemp(dir=r,text=True);os.close(fd);Path(name).write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in rows)+"\n",encoding="utf8");os.replace(name,r/"manifest.jsonl");return True
  elif answer=="r":tmp.unlink(missing_ok=True);return record(r,ident,device,replace=replace)
  elif answer in {"k","q"}:tmp.unlink(missing_ok=True);return False
def replay(r,ident):import winsound;winsound.PlaySound(str(r/"audio"/f"{ident}.wav"),winsound.SND_FILENAME)
def delete(r,ident):
 path=r/"audio"/f"{ident}.wav";print(f"Delete fixture {ident}: {path}");
 if input("Type DELETE to confirm: ")!="DELETE":return
 path.unlink(missing_ok=True);rows=[json.loads(x) for x in (r/"manifest.jsonl").read_text(encoding="utf8").splitlines() if x.strip() and json.loads(x)["id"]!=ident];fd,name=tempfile.mkstemp(dir=r,text=True);os.close(fd);Path(name).write_text("\n".join(json.dumps(x,ensure_ascii=False) for x in rows)+("\n" if rows else ""),encoding="utf8");os.replace(name,r/"manifest.jsonl")
def progress(r):
 p=validate_plan(r);ids={json.loads(x)["id"] for x in (r/"manifest.jsonl").read_text(encoding="utf8").splitlines() if x.strip()};print(f"planned={len(p)} recorded={len(ids)} valid=0 missing={len(p)-len(ids)} invalid=0 remaining={[x['id'] for x in p if x['id'] not in ids]}")
def main():
 p=argparse.ArgumentParser();p.add_argument("action",choices=["init","devices","test-device","list","progress","record","record-next","record-all","replay","re-record","delete","validate"]);p.add_argument("--id");p.add_argument("--device",type=int);p.add_argument("--root",default=".local/parakeet/fixtures/phase_1_5");a=p.parse_args();r=root(a)
 if a.action=="init":init(r)
 elif a.action=="devices":devices()
 elif a.action=="test-device":test_device(a.device)
 elif a.action=="list":print("\n".join(x["id"] for x in validate_plan(r)))
 elif a.action=="progress":progress(r)
 elif a.action=="validate":print(validate_manifest(r))
 elif a.action in {"record","re-record"}:record(r,a.id or "",a.device,replace=a.action=="re-record")
 elif a.action=="record-next":record(r,next(x["id"] for x in validate_plan(r) if not (r/"audio"/f"{x['id']}.wav").exists()),a.device)
 elif a.action=="record-all":
  for x in validate_plan(r):
   if not (r/"audio"/f"{x['id']}.wav").exists() and not record(r,x["id"],a.device):break
 elif a.action=="replay":replay(r,a.id or "")
 elif a.action=="delete":delete(r,a.id or "")
if __name__=="__main__":main()
