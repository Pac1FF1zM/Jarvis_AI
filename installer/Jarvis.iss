#define AppName "Jarvis AI"
#define AppVersion "0.6.0"
#define AppPublisher "Pac1FF1zM"

[Setup]
AppId={{B7EAF4EA-64BE-4BB4-8C92-176D1A29D106}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL=https://github.com/Pac1FF1zM/Jarvis_AI
AppSupportURL=https://github.com/Pac1FF1zM/Jarvis_AI/issues
DefaultDirName={localappdata}\Programs\Jarvis
DefaultGroupName=Jarvis AI
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
DisableProgramGroupPage=yes
WizardStyle=modern
Compression=lzma2/ultra64
SolidCompression=yes
OutputDir=output
OutputBaseFilename=Jarvis_Setup
UninstallDisplayName=Jarvis AI
SetupLogging=yes
CloseApplications=yes

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Создать ярлык Jarvis на рабочем столе"; GroupDescription: "Ярлыки:"; Flags: checkedonce
Name: "ollama"; Description: "Установить Jarvis Full: Ollama и qwen2.5:7b-instruct (несколько ГБ)"; GroupDescription: "Дополнительные компоненты:"; Flags: unchecked

[Dirs]
Name: "{userappdata}\Jarvis"; Flags: uninsneveruninstall
Name: "{userappdata}\Jarvis\logs"; Flags: uninsneveruninstall

[Files]
Source: "..\main.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\jarvis_control.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\config.yaml"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\control_center\*.py"; DestDir: "{app}\control_center"; Flags: ignoreversion
Source: "..\core\*.py"; DestDir: "{app}\core"; Flags: ignoreversion
Source: "..\memory\*.py"; DestDir: "{app}\memory"; Flags: ignoreversion
Source: "..\modules\*.py"; DestDir: "{app}\modules"; Flags: ignoreversion
Source: "..\tools\*.py"; DestDir: "{app}\tools"; Flags: ignoreversion
Source: "..\ml\__init__.py"; DestDir: "{app}\ml"; Flags: ignoreversion
Source: "..\ml\nlu\__init__.py"; DestDir: "{app}\ml\nlu"; Flags: ignoreversion
Source: "..\ml\nlu\inference.py"; DestDir: "{app}\ml\nlu"; Flags: ignoreversion
Source: "..\ml\nlu\models.py"; DestDir: "{app}\ml\nlu"; Flags: ignoreversion
Source: "..\ml\nlu\schema.py"; DestDir: "{app}\ml\nlu"; Flags: ignoreversion
Source: "..\ml\nlu\tokenizer.py"; DestDir: "{app}\ml\nlu"; Flags: ignoreversion
Source: "..\ml\gesture\__init__.py"; DestDir: "{app}\ml\gesture"; Flags: ignoreversion
Source: "..\ml\gesture\labels.py"; DestDir: "{app}\ml\gesture"; Flags: ignoreversion
Source: "..\ml\gesture\models.py"; DestDir: "{app}\ml\gesture"; Flags: ignoreversion
Source: "..\src\__init__.py"; DestDir: "{app}\src"; Flags: ignoreversion
Source: "..\src\data\__init__.py"; DestDir: "{app}\src\data"; Flags: ignoreversion
Source: "..\src\data\transforms.py"; DestDir: "{app}\src\data"; Flags: ignoreversion
Source: "..\src\models\*.py"; DestDir: "{app}\src\models"; Flags: ignoreversion
Source: "..\models\nlu_manager_finetuned.pt"; DestDir: "{app}\models"; Flags: ignoreversion
Source: "..\models\nlu_manager_finetuned.metrics.json"; DestDir: "{app}\models"; Flags: ignoreversion skipifsourcedoesntexist
Source: "..\checkpoints\tsn_resnet18_seed42\best.pt"; DestDir: "{app}\checkpoints\tsn_resnet18_seed42"; Flags: ignoreversion
Source: "..\reports\evaluation_test.json"; DestDir: "{app}\reports"; Flags: ignoreversion
Source: "requirements-lite.txt"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "requirements-full.txt"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "prepare_whisper.py"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "bootstrap_runtime.ps1"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "launchers\Jarvis.cmd"; DestDir: "{app}"; Flags: ignoreversion
Source: "launchers\Jarvis Doctor.cmd"; DestDir: "{app}"; Flags: ignoreversion
Source: "launchers\Enable Jarvis Full.cmd"; DestDir: "{app}"; Flags: ignoreversion
Source: "cache\python-3.12.9-amd64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall

[Icons]
Name: "{group}\Jarvis"; Filename: "{app}\Jarvis.cmd"; WorkingDir: "{app}"
Name: "{group}\Jarvis Runtime Doctor"; Filename: "{app}\Jarvis Doctor.cmd"; WorkingDir: "{app}"
Name: "{group}\Включить Jarvis Full (Ollama)"; Filename: "{app}\Enable Jarvis Full.cmd"; WorkingDir: "{app}"
Name: "{autodesktop}\Jarvis"; Filename: "{app}\Jarvis.cmd"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\Jarvis Doctor.cmd"; Description: "Показать итоговую диагностику"; Flags: postinstall nowait skipifsilent unchecked

[UninstallDelete]
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\models\openai-whisper"

[Code]
procedure RunRequired(const FileName, Parameters, WorkingDirectory, LabelText: String);
var
  ResultCode: Integer;
begin
  WizardForm.StatusLabel.Caption := LabelText;
  if not Exec(FileName, Parameters, WorkingDirectory, SW_SHOW,
    ewWaitUntilTerminated, ResultCode) then
    RaiseException('Не удалось запустить обязательный этап: ' + FileName);
  if ResultCode <> 0 then
    RaiseException(Format('Этап завершился с кодом %d: %s', [ResultCode, FileName]));
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  PythonParams, BootstrapParams: String;
begin
  if CurStep <> ssPostInstall then
    Exit;

  PythonParams := '/quiet InstallAllUsers=0 TargetDir="' +
    ExpandConstant('{app}\runtime\python') +
    '" Include_pip=1 Include_launcher=0 Include_test=0 Include_doc=0' +
    ' Shortcuts=0 AssociateFiles=0 PrependPath=0';
  RunRequired(
    ExpandConstant('{tmp}\python-3.12.9-amd64.exe'),
    PythonParams,
    ExpandConstant('{tmp}'),
    'Установка изолированного Python 3.12...');

  BootstrapParams := '-NoProfile -ExecutionPolicy Bypass -File "' +
    ExpandConstant('{app}\installer\bootstrap_runtime.ps1') +
    '" -AppDir "' + ExpandConstant('{app}') +
    '" -DataDir "' + ExpandConstant('{userappdata}\Jarvis') + '"';
  if WizardIsTaskSelected('ollama') then
    BootstrapParams := BootstrapParams + ' -InstallOllama';
  RunRequired(
    ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe'),
    BootstrapParams,
    ExpandConstant('{app}'),
    'Установка Jarvis Lite и выбранных компонентов...');
end;
