# Build the NVIDIA GPU library of Kitten TTS 2 mini, nano and pico for Windows (kt08_cuda.dll: the engine with its CUDA
# backend, the CUDA 12.8 runtime and the C runtime linked in statically, no cuBLAS), shipped in the Kitten TTS 2 CUDA
# payload wheel (kitten_inference/_tts2_cuda/lib/) by engines/tts2/packaging/build_wheel_windows.ps1 -ExtraLib.
#   powershell -File engines\tts08\cuda-library\build_windows.ps1 [-Out DIR] [-Jobs N]
# Run it from a Visual Studio 2022 x64 developer environment with CUDA 12.8 (CUDA_PATH), CMake and Ninja on PATH, as
# the Kitten TTS 2 engine build. The DLL must import only Windows system DLLs (the driver's nvcuda.dll is loaded at run
# time), so it runs on a Windows with only the NVIDIA driver.
param(
  [string]$Out = "",
  [int]$Jobs = 0,
  [string]$Archs = "61-real;75-real;80-real;86-real;89-real;90-real;100-real;120"
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
if (-not $Out) { $Out = Join-Path $Root "engines\tts08\dist" }
if ($Jobs -le 0) { $Jobs = [Environment]::ProcessorCount }
New-Item -ItemType Directory -Force -Path $Out | Out-Null
function Run([string]$exe, [string[]]$argv) {
  & $exe @argv
  if ($LASTEXITCODE -ne 0) { throw "$exe $($argv -join ' ') failed with exit code $LASTEXITCODE" }
}
$Build = Join-Path $Root "engines\tts08\build\cuda-library-windows"
$Stage = "$Build-stage"
foreach ($d in @($Build, $Stage)) { if (Test-Path $d) { Remove-Item -Recurse -Force $d } }
Run "cmake" @("-S", (Join-Path $Root "engines\tts08\engine"), "-B", $Build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
              "-DCMAKE_C_COMPILER=cl", "-DCMAKE_CXX_COMPILER=cl", "-DENABLE_CUDA=ON", "-DKITTEN_BUILD_CUDA_LIBRARY=ON",
              "-DCMAKE_CUDA_ARCHITECTURES=$Archs", "-DKITTEN_BUILD_PYTHON_MODULE=OFF", "-DENABLE_NATIVE_ARCH=OFF",
              "-DENABLE_X86_AVX2=ON", "-DENABLE_ARM_DOTPROD=OFF", "-DENABLE_OPENMP=OFF", "-DKITTEN_BUILD_EXTRAS=OFF")
Run "cmake" @("--build", $Build, "--target", "kt08_cuda", "--parallel", "$Jobs")
Run "cmake" @("--install", $Build, "--component", "cuda_library", "--prefix", $Stage)
$dll = Join-Path $Stage "bin\kt08_cuda.dll"
$allowed = @("KERNEL32.dll", "ADVAPI32.dll", "WINMM.dll", "SHELL32.dll", "USER32.dll", "ole32.dll", "OLEAUT32.dll",
             "WS2_32.dll", "bcrypt.dll", "SHLWAPI.dll", "VERSION.dll", "dbghelp.dll")
$all = (& dumpbin /nologo /dependents $dll) | Where-Object { $_ -match '^\s+\S+\.dll\s*$' } |
  ForEach-Object { $_.Trim() } | Sort-Object -Unique
$bad = $all | Where-Object { $allowed -notcontains $_ }
"kt08_cuda.dll imports: $($all -join ', ')"
if ($bad) { throw "kt08_cuda.dll imports non-system DLLs: $($bad -join ', ')" }
Copy-Item $dll (Join-Path $Out "kt08_cuda.dll") -Force
"$(Join-Path $Out 'kt08_cuda.dll') $((Get-Item $dll).Length) bytes"
