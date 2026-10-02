$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$outputRoot = Join-Path $projectRoot 'dist'
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$archive = Join-Path $outputRoot 'StrataCuda-WebCuda.zip'
$items = @('index.html','test.html','web','src','kernels','vendor','generated','scripts','tests','reports',
    'README.md','LICENSE','THIRD_PARTY_NOTICES.md','PROVENANCE.json','package.json','package-lock.json','requirements-model.txt','START.bat')
$paths = $items | ForEach-Object { Join-Path $projectRoot $_ }
Compress-Archive -LiteralPath $paths -DestinationPath $archive -Force -CompressionLevel Optimal
$algorithm = [Security.Cryptography.SHA256]::Create()
$stream = [IO.File]::OpenRead($archive)
try { $hash = -join ($algorithm.ComputeHash($stream) | ForEach-Object { $_.ToString('x2') }) }
finally { $stream.Dispose(); $algorithm.Dispose() }
[IO.File]::WriteAllText((Join-Path $outputRoot 'SHA256SUMS.txt'), "$hash  StrataCuda-WebCuda.zip`n")
Write-Output "$archive ($((Get-Item -LiteralPath $archive).Length) bytes)"
Write-Output "SHA256 $hash"
