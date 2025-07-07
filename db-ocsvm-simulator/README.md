# db-ocsvm-simulator

## Overview

db-ocsvm-simulator is a desktop and web application for evaluating and visualizing the performance of machine learning models, specifically One-Class SVM (OCSVM) and DBSCAN-OCSVM, on network intrusion datasets. The simulator provides an interactive interface for running inference, visualizing results, and comparing models.

## Features

- Desktop app (Electron) and web app (React)
- Visualize and compare OCSVM and DBSCAN-OCSVM results
- Load and process datasets (e.g., CIDDS-001, NSL-KDD)
- Run inference using pre-trained ONNX and joblib models
- Modern UI with Tailwind CSS
- Modular codebase for easy extension

## Folder Structure

```
db-ocsvm-simulator/
├── src/
│   ├── main/           # Electron main process code
│   ├── renderer/       # React renderer process code
│   └── ...
├── index.html          # Main HTML entry point
├── package.json        # Project metadata and scripts
├── electron-builder.json # Electron build config
├── tailwind.config.js  # Tailwind CSS config
├── vite.config.ts      # Vite build config
└── ...
```

## Setup

1. **Clone the repository**

   ```sh
   git clone git@github.com:kenBinary/db-ocsvm-simulator.git
   cd db-ocsvm-simulator
   ```

2. **Install dependencies**
   ```sh
   npm install
   ```

## Usage

- **Run the desktop app (Electron + React):**

  ```sh
  npm run dev
  ```

- **Run only the web app (React):**

  ```sh
  npm run dev:react
  ```

- **Build the React app:**

  ```sh
  npm run build:react
  ```

- **Transpile Electron code:**
  ```sh
  npm run transpile:electron
  ```

## Troubleshooting

- Ensure Node.js and npm are installed (Node.js >= 16 recommended).
- If you encounter dependency issues, try deleting `node_modules` and `package-lock.json`, then reinstall:
  ```sh
  rm -rf node_modules package-lock.json
  npm install
  ```
- For Electron issues, ensure you are running the correct Node version and have all build tools installed.
