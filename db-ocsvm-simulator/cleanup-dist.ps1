# Script to delete dist-related folders

$foldersToDelete = @("dist", "dist-electron", "dist-react")
$currentPath = Get-Location

foreach ($folder in $foldersToDelete) {
    $folderPath = Join-Path -Path $currentPath -ChildPath $folder
    
    if (Test-Path -Path $folderPath) {
        Write-Host "Deleting folder: $folder"
        Remove-Item -Path $folderPath -Recurse -Force
    } else {
        Write-Host "Folder not found: $folder"
    }
}

Write-Host "Cleanup complete!"