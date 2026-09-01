Get-ChildItem 'C:\tools\cudnn\bin\' -Filter '*.bak' | ForEach-Object {
    $newName = $_.Name -replace '\.bak$', ''
    Rename-Item -Path $_.FullName -NewName $newName
    Write-Output ("Renamed: " + $_.Name + " -> " + $newName)
}
