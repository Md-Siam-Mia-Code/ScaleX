<p align="center">
  <img src="assets/Banner.png" alt="ScaleX Web Banner">
</p>

# 🌟 ScaleX WebUI: Browser-Based AI Face Magic! ✨🖼️

No CLI needed! ScaleX WebUI puts AI face restoration & image enhancement in your web browser. Powered by GFPGAN & Real-ESRGAN, now with a slick UI, live previews, & easy controls! 🚀

## 🗂 Menu
- 📖 [What's This?](#-whats-this)
- ✨ [Awesome Features!](#-awesome-features)
- 🛠️ [Installation](#️-installation)
  - 📋 [You Need](#-you-need)
  - 🚀 [Auto-Install (Easy!)](#-auto-install-easy)
  - 🔩 [Manual Setup](#-manual-setup)
- 💻 [Let's Go! (Usage)](#️-lets-go-usage)
  - ▶️ [Launch UI](#️-launch-ui)
  - 🎨 [Using It](#-using-it)
- 💡 [Help! (Troubleshooting)](#-help-troubleshooting)
- 🤝 [Contribute!](#-contribute)
- 📜 [Credits & License](#-credits--license)

---

## 📖 What's This?
ScaleX WebUI transforms your old, blurry photos right in your browser! 🪄 It uses AI (GFPGAN & Real-ESRGAN) to make faces pop ✨ and backgrounds beautiful 🏞️. Super easy, super fast!

---

## ✨ Awesome Features!
- 🚀 **AI Face Restore:** GFPGAN v1.3 & v1.4 for stunning faces.
- 🖼️ **BG Enhancement:** Real-ESRGAN (x2/x4) for crisp backgrounds.
- 📈 **Adjustable Upscaling:** You control the final size! 🐘
- 😍 **Slick Web UI:** Modern, responsive, and fun to use.
- 📤 **Drag & Drop Upload:** Easy image handling.
- 🖼️ **Live Progress & Previews:** Watch the magic unfold! 🪄
- 🌓 **Comparison Slider:** Original vs. Enhanced, side-by-side.
- ⚙️ **Advanced Controls:** BG Tile, Format, Fidelity, & more!
- 💾 **Save Options:** Full image, cropped/restored faces, comparisons.
- 🎯 **Pre-Aligned Support:** Got 512x512 face crops? Covered! 👍
- 💪 **Device Choice:** Auto/CPU/CUDA/MPS. Use your power! ⚡
- 📂 **Custom Output Folder:** Save where you want.
- 🗑️ **Cache Clearing:** Keep things tidy.
- 📱 **Responsive Design:** Works on Desktop, Tablet, Mobile! 💻📱
- 🔧 **Auto-Patches:** Bye-bye `torchvision` issues! 👋

---

## 🛠️ Installation
### 📋 You Need
- 🐍 Python 3.10 - 3.12 (3.12 recommended!)
- 🐉 Anaconda/Miniconda (For easy env management)
- ➕ Git
- ❗ **Optional GPU Power:**
  - NVIDIA GPU: Latest [CUDA Drivers](https://www.nvidia.com/Download/index.aspx).
  - Apple Silicon (M-Series): macOS 12.3+.

### 🚀 Auto-Install (Easy!)
Let's get it done! 🥳
1.  **Clone Repo:**
    ```bash
    git clone https://github.com/Md-Siam-Mia-Code/ScaleX.git
    cd ScaleX
    ```
2.  **Run Installer:** ✨
    *   **Windows 🧙‍♂️:** (In `ScaleX` folder)
        ```batch
        install_scalex_windows.bat
        ```
    *   **Linux/macOS 🥷:** (In `ScaleX` folder)
        ```bash
        chmod +x install_scalex_linux.sh
        ./install_scalex_linux.sh
        ```
    Follow the script's prompts! 🕺

### 🔩 Manual Setup
For the brave! 🤠
1.  **Clone Repo** (if not done).
2.  **Conda Environment:**
    ```bash
    conda create -n ScaleXWeb python=3.12 -y
    conda activate ScaleXWeb
    ```
3.  **Install PyTorch:** From [PyTorch website](https://pytorch.org/get-started/locally/). Choose Conda, Python, and your Compute Platform (CUDA/CPU/MPS).
4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Models:** Auto-downloaded on first use!  моделей ✨ (Usually into `ScaleX - Web/models/pretrained/`)

---

## 💻 Let's Go! (Usage)
### ▶️ Launch UI
1.  Open Terminal/Anaconda Prompt.
2.  Activate env: `conda activate ScaleXWeb`
3.  Go to `ScaleX - Web` sub-directory: `cd "ScaleX - Web"`
4.  Run: `python webui_scalex.py`
5.  Open browser to: `http://127.0.0.1:5000` 🌐

### 🎨 Using It
1.  📤 **Upload:** Drag & drop or click to select an image.
2.  ⚙️ **Settings:** Adjust basic/advanced options in the sidebar.
3.  🚀 **Process:** Click "ScaleX It!"
4.  📊 **Monitor:** Watch progress bar & status. Logs in Settings Panel.
5.  🖼️ **Results:**
    *   Enhanced image appears.
    *   Hover for: **Compare** (slider 🌓), **Download** 💾, **Re-Process** 🔄.
6.  🛠️ **Settings Panel:** Click <i class="fas fa-cog"></i> icon.
    *   Sidebar turns into settings nav (Logs, Themes, System Info, etc.).
    *   Main panel shows selected settings content.
    *   Click <i class="fas fa-arrow-left"></i> to go back.

---

## 💡 Help! (Troubleshooting)
*   **`functional_tensor` Error?** `patches.py` should auto-fix. ✅
*   **Model Download Stuck?** 🐌 Check internet. Delete partial `.pth` in `ScaleX - Web/models/pretrained/` & retry.
*   **GPU Not Working?** 🙅‍♀️
    *   NVIDIA: Drivers updated? PyTorch CUDA version correct?
    *   Apple: macOS 12.3+?
    *   Try "CPU" in Device settings.
*   **Install Script Failed?** Make sure Conda is in PATH.
*   **`ModuleNotFoundError`?** Are you in the `ScaleXWeb` conda env? `pip install -r requirements.txt` successful?

---

## 🤝 Contribute!
Got ideas? Found bugs 🐞? Want to add sparkle ✨? Contributions welcome!
Check [issues](https://github.com/Md-Siam-Mia-Code/ScaleX/issues) or send a PR!
1.  🍴 Fork
2.  🌿 New Branch (`git checkout -b feature/CoolStuff`)
3.  💾 Commit (`git commit -m 'Added CoolStuff'`)
4.  🚀 Push (`git push origin feature/CoolStuff`)
5.  📬 Open PR!

---

## 📜 Credits & License
Built on the shoulders of giants! 🏋️‍♂️ Thanks to:
*   **GFPGAN:** [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN)
*   **Real-ESRGAN:** [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
*   **BasicSR & facexlib:** [xinntao](https://github.com/xinntao)
*   And many other great libraries! 📚

---

# ❤️ *Make Your Pixels Shine!* ✨😄
