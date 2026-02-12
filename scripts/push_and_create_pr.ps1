<#
.SYNOPSIS
  Configure repo remote (HTTPS), push branch main, and create a PR using gh.

.DESCRIPTION
  Safe workflow that does NOT accept tokens in plaintext. Uses Git credential manager
  for HTTPS auth (will prompt) and GitHub CLI (gh) to create the PR interactively.

Usage:
  .\scripts\push_and_create_pr.ps1 -Title "Initial import" -Body "Add project scaffold" 

Requires:
  - Git installed
  - Git Credential Manager (usually bundled with Git for Windows)
  - GitHub CLI `gh` installed and logged in (script will prompt to login if necessary)
#>
param(
  [string]$RemoteHttps = "https://github.com/Arjunlakhanpall/Neurolinked.git",
  [string]$Title = "Initial import",
  [string]$Body = "Add project scaffold, scripts, notebooks, and CI.",
  [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

function Exec($cmd) {
  Write-Host "> $cmd"
  Invoke-Expression $cmd
}

# Check git
try {
  Exec "git --version"
} catch {
  Write-Error "Git not found. Install Git and re-run: https://git-scm.com/downloads"
  exit 1
}

# Configure remote to HTTPS
try {
  if ((git remote) -contains "origin") {
    Exec "git remote set-url origin $RemoteHttps"
  } else {
    Exec "git remote add origin $RemoteHttps"
  }
} catch {
  Write-Error "Failed to set remote: $_"
  exit 1
}

# Ensure credential helper (manager-core) to prompt for credentials
try {
  Exec "git config --global credential.helper manager-core"
} catch {
  Write-Warning "Failed to configure credential.helper; you may be prompted for credentials on push."
}

# Add and commit changes if any
try {
  Exec "git add ."
  $status = git status --porcelain
  if ($status) {
    $msg = Read-Host "Enter commit message (default: '$Title')"
    if (-not $msg) { $msg = $Title }
    Exec "git commit -m `"$msg`""
  } else {
    Write-Host "No changes to commit."
  }
} catch {
  Write-Warning "Commit step failed or no changes: $_"
}

# Ensure branch name
try {
  Exec "git branch -M $Branch"
} catch {
  Write-Warning "Failed to set branch to $Branch: $_"
}

# Ensure gh is installed and authenticated
try {
  Exec "gh --version"
} catch {
  Write-Error "GitHub CLI (gh) not found. Install from https://cli.github.com/ and re-run the script."
  exit 1
}

try {
  $auth = gh auth status 2>&1
  if ($LASTEXITCODE -ne 0) {
    Write-Host "You are not logged in with gh. Running 'gh auth login' now..."
    Exec "gh auth login"
  } else {
    Write-Host "gh is authenticated."
  }
} catch {
  Write-Error "gh authentication failed: $_"
  exit 1
}

# Push to remote
try {
  Exec "git push -u origin $Branch"
} catch {
  Write-Error "Push failed. Check credentials and remote access: $_"
  exit 1
}

# Create PR (interactive if needed)
try {
  Exec "gh pr create --title `"$Title`" --body `"$Body`" --base $Branch"
} catch {
  Write-Error "Failed to create PR via gh: $_"
  exit 1
}

Write-Host "Done. PR created (or opened interactively)."
