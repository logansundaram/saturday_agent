# Embedded Qdrant Binaries

Place platform binaries in the following paths:

- `macos-arm64/qdrant`
- `macos-x64/qdrant`
- `win-x64/qdrant.exe` (or `qdrant` if pre-renamed)
- `linux-x64/qdrant`

The Electron main process launches this binary with:

- `--storage-path <userData>/qdrant/storage`
- `--http-port <dynamic-port>`
