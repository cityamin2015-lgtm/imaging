#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

[Setup]
AppName=Pattern Engine
AppVersion={#AppVersion}
AppPublisher=Pattern Engine
DefaultDirName={autopf}\PatternEngine
DefaultGroupName=Pattern Engine
OutputDir=installer_output
OutputBaseFilename=PatternEngine-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
UninstallDisplayIcon={app}\PatternEngine.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "dist\PatternEngine.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Pattern Engine"; Filename: "{app}\PatternEngine.exe"
Name: "{group}\Uninstall Pattern Engine"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Pattern Engine"; Filename: "{app}\PatternEngine.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\PatternEngine.exe"; Description: "Launch Pattern Engine"; Flags: nowait postinstall skipifsilent
