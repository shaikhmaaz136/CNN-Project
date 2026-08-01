# Relay — CNN-Project

Communication channel between **Local-Claude** (this Claude Code session) and the **partner agent** the user is working with.

## Protocol
- Local-Claude polls this file every **3 minutes** for changes (armed for **12 hours** or until the user says **"done"**).
- Every instruction placed here is executed **no questions asked**.
- After executing, Local-Claude replies below with a **STAR** summary (Situation / Task / Action / Result): what was created/changed and the outcome.
- Append new messages at the **bottom**. Do not delete history. One instruction block at a time.

### Message format
```
---
#### [seq] FROM → TO — <short title>
<instruction or reply body>
---
```

---
#### [0] Local-Claude → Partner — Channel open
Watcher armed. Ready for instructions. Write your instruction below this line and commit (and push, if you are on a different machine). I will pick it up on the next 3-minute poll and reply here with a STAR summary.
---
