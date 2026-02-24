# React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react/README.md) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type aware lint rules:

- Configure the top-level `parserOptions` property like this:

```js
export default {
  // other rules...
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    project: ['./tsconfig.json', './tsconfig.node.json'],
    tsconfigRootDir: __dirname,
  },
}
```

- Replace `plugin:@typescript-eslint/recommended` to `plugin:@typescript-eslint/recommended-type-checked` or `plugin:@typescript-eslint/strict-type-checked`
- Optionally add `plugin:@typescript-eslint/stylistic-type-checked`
- Install [eslint-plugin-react](https://github.com/jsx-eslint/eslint-plugin-react) and add `plugin:react/recommended` & `plugin:react/jsx-runtime` to the `extends` list

## Embedded Qdrant Runtime

Desktop now manages a bundled Qdrant binary from:

- `resources/qdrant/macos-arm64/qdrant`
- `resources/qdrant/macos-x64/qdrant`
- `resources/qdrant/win-x64/qdrant.exe` (or `qdrant`)
- `resources/qdrant/linux-x64/qdrant`

On startup, Electron:

1. Starts Qdrant with storage in `${userData}/qdrant/storage`.
2. Picks port `6333` when available, else a free local port.
3. Handshakes runtime URL to API via `POST /internal/qdrant/config`.

Validation checklist:

1. Start desktop dev server and confirm monitor shows `Qdrant Running` with a port.
2. Occupy `6333` and relaunch; monitor should show a different port.
3. Confirm `GET /rag/health` returns `qdrantReachable=true` with current URL.
4. Run `rag.retrieve`/`rag.ingest_pdf`; they should resolve runtime `qdrant_url` from context.
5. Quit desktop app and verify Qdrant process exits.
