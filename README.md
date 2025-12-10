# ESW_PreBorn — Project README

> A concise project guide for the ESW_PreBorn workspace: what we built, how to run it, and how to contribute.

---

**Table of contents**

- Project overview
- Prerequisites
- Installation & local setup
- Firebase setup & explanation
- How to use (run / build / deploy)
- What we have done (features)
- How to enhance (ideas & roadmap)
- Contributing & raising issues
- Supporting open source
- Contributors
- License & contact

---

## Project overview

This repository contains the ESW_PreBorn project — a personal / course website and related tools (e.g., Text Analyzer, Projects showcase, small web-apps and utilities). The goal is to collect, present, and demonstrate practical projects (web, CV/AI, IoT, systems) and provide utilities used during development.

This README documents how to set up the project locally, how Firebase is used, what has already been implemented, and how others can contribute.

## Prerequisites

Install these on your development machine before continuing:

- Git (>= 2.x)
  - https://git-scm.com/
- Node.js & npm (LTS recommended, e.g., Node 18+)
  - https://nodejs.org/
- Python 3.8+ (only if running Python tools or scripts)
  - https://www.python.org/
- Firebase CLI (if you plan to use Firebase hosting, functions, or emulators)
  - Install: `npm install -g firebase-tools`
  - Authenticate: `firebase login`
- (Optional) A modern browser (Chrome/Firefox/Edge) for local testing

Notes about environment:
- Development was done on Linux; commands below assume bash.
- If you use virtual environments for Python, create one before installing Python deps.

## Installation & local setup

1. Clone the repo (example path/name):

```bash
git clone https://github.com/<your-username>/ESW_PreBorn.git
cd ESW_PreBorn
```

2. Install Node dependencies (if the project contains a frontend with package.json):

```bash
# from repo root, if package.json exists
npm install
```

3. If there are Python requirements:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Configure environment variables (see `.env.example` if available). Common env vars:

```
# Example .env
FIREBASE_API_KEY=...
FIREBASE_AUTH_DOMAIN=project-id.firebaseapp.com
FIREBASE_PROJECT_ID=project-id
FIREBASE_STORAGE_BUCKET=project-id.appspot.com
FIREBASE_MESSAGING_SENDER_ID=...
FIREBASE_APP_ID=...
```

5. Open the project in your editor (VS Code recommended).

## Firebase setup & explanation

This project uses Firebase for one or more of the following services:
- Hosting — serve static site (e.g., the personal website)
- Firestore / Realtime Database — store app data (e.g., depth maps, user submissions)
- Storage — store assets (images, models, DLC files)
- Authentication — optional user sign-in
- Cloud Functions — server-side logic (optional)

How to create and connect a Firebase project:

1. Create a Firebase project at https://console.firebase.google.com/.
2. From the project overview, add a web app to get the Firebase config object (API key, projectId, etc.).
3. Add the config to your frontend application as environment variables or directly in a secure config file.

Example client-side initialization (web):

```javascript
import { initializeApp } from 'https://www.gstatic.com/firebasejs/9.x.x/firebase-app.js';
import { getFirestore } from 'https://www.gstatic.com/firebasejs/9.x.x/firebase-firestore.js';

const firebaseConfig = {
  apiKey: process.env.FIREBASE_API_KEY,
  authDomain: process.env.FIREBASE_AUTH_DOMAIN,
  projectId: process.env.FIREBASE_PROJECT_ID,
  storageBucket: process.env.FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.FIREBASE_APP_ID
};

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
```

Using Firebase CLI for hosting & functions:

```bash
# Initialize Firebase in the project (one-time)
firebase init

# To run local emulators (recommended while developing functions / DB rules)
firebase emulators:start

# To deploy hosting & functions
firebase deploy --only hosting
# or for functions
firebase deploy --only functions
```

Security & rules:
- Define Firestore or Realtime DB security rules early and test them with the emulator.
- Use storage rules to prevent public write access unless intended.

## How to use (run / build / deploy)

Local dev (simple static site):

```bash
# If site is static, a simple way is to use a local static server
npx http-server . -p 8080
# then open http://localhost:8080/index.html
```

If a Node-based dev server exists (e.g., React/Vue):

```bash
npm run dev
# or
npm start
```

Build and deploy (static):

```bash
npm run build    # if applicable
# Then deploy to Firebase hosting
firebase deploy --only hosting
```

Run Python services (if present):

```bash
# example for Flask app
export FLASK_APP=app.py
export FLASK_ENV=development
flask run
```

Verify features manually in the browser:
- Homepage (`index.html`)
- Projects (`Projects.html`)
- Text Analyzer (`TextAnalyzer.html`)

## What we have done (features)

This project currently includes:

- Personal website layout and stylistic theme (hand-drawn / sketch look with mint green accents)
- Main homepage with interactive elements: draggable notebook, paper plane (CV) preview, background audio controls
- `Projects.html` — a flashcard-style projects showcase (project name, objective, description, tech and links)
- `TextAnalyzer.html` — standalone text analysis tool (word/character counts, readability hints, sentiment insights)
- `script.js` — site interactivity logic and linking of the notebook to `Projects.html`
- `styles.css` — main site styling matching the theme
- Basic Firebase-ready structure (config placeholders) for hosting, storage, and real-time features

If any of the above lives in separate directories, check: `index.html`, `Projects.html`, `TextAnalyzer.html`, `js/`, `images/`, and `audio/`.

## How to enhance (ideas & roadmap)

Short-term improvements:
- Add issue templates and PR templates to standardize contributions
- Add `CODE_OF_CONDUCT.md` and `CONTRIBUTING.md`
- Add unit tests for JavaScript utilities (Jest) and smoke tests for pages (Playwright)
- Add lazy-loading for images and optimize assets
- Make the site responsive improvements for small screens

Medium-term features:
- Add a small CMS (e.g., Netlify CMS or a simple Firebase-backed admin) for adding projects without editing HTML
- Integrate analytics (privacy-respecting) to track page visits
- Add a searchable filter on `Projects.html` (by tag / tech)
- Add automated CI to lint/build and deploy to Firebase (GitHub Actions)

Long-term:
- Turn text analyzer into a small progressive web app (PWA) with offline support
- Add user accounts and allow users to save analysis sessions (requires Firebase Auth + Firestore)
- Add internationalization (i18n)

## Contributing & raising issues

We welcome contributions. Suggested workflow:

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/short-description`
3. Make your changes with clear, atomic commits
4. Push branch to your fork and open a Pull Request against `main` with a clear description and link to any related issue

Raising issues:
- Use GitHub Issues to report bugs, request features, or ask questions
- Provide a clear title and description, expected vs actual behavior, steps to reproduce, and screenshots/logs if available

Example issue template fields (suggested):
- Title
- Description
- Steps to reproduce
- Expected behavior
- Environment (browser, OS)
- Screenshots / logs

Pull request checklist (example):
- [ ] Changes documented in README (if behavior or setup changed)
- [ ] Code is linted (run `npm run lint` if available)
- [ ] No sensitive keys in the changes

## Supporting open source

If you find this project useful, you can support it by:
- Starring or forking the repo on GitHub
- Opening issues and reporting bugs
- Submitting PRs with improvements
- Sharing the project with others

If you'd like to sponsor development or need a feature built, contact the maintainer (see Contact section below).

## Contributors

This project is maintained by:

- Gyandeep Das — primary author
- Harshil Soni — primary author 
- Samarth Jain — primary author
- Swayam Goyal — primary author

## License & contact

- No License. Free to use for the betterment of the Open source society.

Contact:
- GitHub: https://github.com/MaxGD012 or https://github.com/gyandeepdas
- Webpage - gyandeepdas.github.io

---

## Quick commands summary

```bash
# clone
git clone <repo>
cd ESW_PreBorn

# run local static server
npx http-server . -p 8080
# open http://localhost:8080/index.html

# firebase (deploy)
firebase login
firebase init      # configure hosting/functions if needed
firebase deploy --only hosting
```
