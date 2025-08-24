# Scripts

Development and automation scripts for the Cua project.

## Build Scripts

- **`build.sh`** - Main build script that sets up Python development environment with pip
- **`build-uv.sh`** - Alternative build script using the UV package manager (faster)
- **`build.ps1`** - PowerShell build script for Windows development

## Playground Scripts

- **`playground.sh`** - Interactive launcher for the Cua Computer-Use Agent UI (choose between cloud containers or local VMs)
- **`playground-docker.sh`** - Docker-based playground launcher
- **`run-docker-dev.sh`** - Development environment runner using Docker

## Utility Scripts

- **`cleanup.sh`** - Removes build artifacts, caches, and virtual environments

## Usage

### Setting up development environment
```bash
# Using pip (standard)
./scripts/build.sh

# Using UV (faster)
./scripts/build-uv.sh

# On Windows
./scripts/build.ps1
```

### Running the playground
```bash
./scripts/playground.sh
```

### Cleaning up
```bash
./scripts/cleanup.sh
```

All scripts should be run from the project root directory.