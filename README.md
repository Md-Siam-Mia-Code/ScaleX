<p align="center">
  <img src="assets/Banner.png" alt="ScaleX Banner">
</p>

# 🌟 ScaleX: ✨ AI Face Magic Wand ✨ (CLI Edition!)

Zap those pixels! ScaleX is your command-line buddy for making faces look ✨ FABULOUS ✨ using AI smarts. Built on cool tech like GFPGAN & Real-ESRGAN, but now with Python 3.12 super-speed, a slick CLI, and zero setup headaches!

## 🗂 Quick Guide Menu
- 📖 [WTF is ScaleX?](#-wtf-is-scalex)
- ✨ [Super Powers!](#-super-powers)
- 🛠️ [Get This Thing Installed!](#️-get-this-thing-installed)
  - 📋 [Stuff You Need First](#-stuff-you-need-first)
  - 🚀 [Easy Peasy Auto-Install (Recommended!)](#-easy-peasy-auto-install-recommended)
  - 🔩 [DIY Manual Setup](#-diy-manual-setup)
- 💻 [Let's Cook! (Usage)](#️-lets-cook-usage)
  - ▶️ [Fire It Up!](#️-fire-it-up)
  - ⚙️ [Magic Spells (Options)](#️-magic-spells-options)
- 💡 [Uh Oh! (Troubleshooting)](#-uh-oh-troubleshooting)
- 🤝 [Join the Fun! (Contributing)](#-join-the-fun-contributing)
- 📜 [The Fine Print (Acknowledgments & License)](#-the-fine-print-acknowledgments--license)

---

## 📖 What is ScaleX?
ScaleX turbocharges your old, blurry, or just plain sad-looking photos right from your terminal! 命令行里的魔法棒! 🪄 It uses AI brains (🧠 GFPGAN & Real-ESRGAN) to make faces pop and backgrounds beautiful. This ain't your grandma's photo editor (unless she's a Python 3.12 coding wizard with a slick CLI 😉). We've made it easy, fast, and fixed those annoying bugs for ya! 🐞➡️🦋

---

## ✨ Super Powers!
- 🚀 **Face Makeover 9000:** GFPGAN v1.3 & v1.4 for A+ faces!
- 🖼️ **Background Bling:** Real-ESRGAN x2/x4 for spiffy backdrops! ✨
- 📈 **Go BIG!** You control the final image size.  ใหญ่ขึ้น!
- 💻 **Friendly CLI:** So easy, your cat could probably use it. 😼
- 🗂 **One or Many:** Single pics or whole albums - BAM! 💥
- 🎨 **Get ALL The Pixels:** Restored images, cropped faces, comparisons - you name it!
- 🎯 **Already Aligned?** Got 512x512 faces? We gotchu. 👍
- ⚙️ **Device Freedom!** CPU 🐢, CUDA 🚀, MPS 🍎 - your choice!
- 📝 **Your Style:** Custom file names & types.
- 📊 **Cool Progress Bars:** Watch the magic happen!  ProgressBarGoBrrr.gif
- 🔧 **Auto-Fixer-Upper:** No more weird `torchvision` drama! 😌

---

## 🛠️ Get This Thing Installed!
### 📋 Stuff You Need First
- 🐍 Python 3.12 (The slithery kind!)
- 🐉 Anaconda/Miniconda (The big snake's cousin, for easy setup!)
- ➕ Git (To grab the code!)
- ❗ **Want SPEED? (Optional GPU Fun):**
  - NVIDIA GPU + Fresh [CUDA Drivers](https://www.nvidia.com/Download/index.aspx) 쌩쌩 달려요!

### 🚀 Easy Peasy Auto-Install (Recommended!)
Get ready for an install party! 🥳
1.  **Yoink the Code!**
    ```bash
    git clone https://github.com/Md-Siam-Mia-Code/ScaleX.git
    cd ScaleX
    ```
2.  **Run the Magic Script!** ✨ (It does almost everything for you!)
    *   **Windows Wizards 🧙‍♂️:**
        Open Command Prompt/PowerShell in `ScaleX` folder, then:
        ```batch
        install_scalex_windows.bat
        ```
    *   **Linux/macOS Ninjas 🥷:**
        Open Terminal in `ScaleX` folder, then:
        ```bash
        chmod +x install_scalex.sh
        ./install_scalex_linux.sh
        ```
    Follow the script's groovy instructions! 🕺💃

### 🔩 DIY Manual Setup
Feeling brave? Like to live dangerously? 🤠
1.  **Clone It (Again, if you skipped above):**
    ```bash
    git clone https://github.com/Md-Siam-Mia-Code/ScaleX.git
    cd ScaleX
    ```
2.  **Conda Cave:**
    ```bash
    conda create -n ScaleX python=3.12 -y
    conda activate ScaleX
    ```
3.  **Get PyTorch (The Brains!):**
    Go to [PyTorch website](https://pytorch.org/get-started/locally/). Pick your poison (OS, Conda, Python, CUDA/CPU).

    *CPU Crew (conda):*
    ```bash
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    ```
    *CUDA Crew (conda, e.g.):*
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=<your-cuda-version> -c pytorch -c nvidia -y
    ```
    _(Change `<your-cuda-version>` with your actual CUDA version!)_

4.  **Install the Rest of the Goodies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Models? We Got 'Em!**
    🚀 They'll download themselves with cool progress bars when you first run ScaleX! ✨ (Usually into `models/pretrained/`)

---

## 💻 Let's Cook! (Usage)
### ▶️ Fire It Up!
Ready to make some magic? 🪄
1.  Open your terminal/Anaconda Prompt.
2.  Warp into the environment: `conda activate ScaleX`
3.  Zoom to the `ScaleX` folder.
4.  Let 'er rip: `python inference_scalex.py ...`

**The Secret Incantation:**
```bash
python inference_scalex.py --input <your_pics_here> --output <where_the_magic_goes> [MORE_SPELLS]
```

**Example - Full Power!**
Make `MyPhotos/` awesome, save to `EnhancedPhotos/`, use GFPGAN v1.4, RealESRGAN x4 for BG, and 4x bigger!
```bash
python inference_scalex.py -i MyPhotos/ -o EnhancedPhotos/ -f v1.4 -b x4 -s 4
```

### ⚙️ Magic Spells (Options)
*   `-i, --input PATH`: **REQUIRED!** Your pics (one or a folder). 🖼️
*   `-o, --output PATH`: Where the awesomeness lands. (Default: `results_scalex`)
*   `-f, --face-enhance [v1.3|v1.4]`: Face fixer model. (Default: `v1.4`) ✨
*   `-b, --bg-enhance [none|x2|x4]`: Background booster. `none` for no boost. (Default: `x2`) 🌳
*   `-s, --upscale INTEGER`: Make the whole pic BIGGER! (Default: `2`) 🐘
*   `--bg-tile INTEGER`: Memory saver for BG. 0 = YOLO mode (faster if you got RAM). (Default: `400`)
*   `--device [auto|cpu|cuda|mps]`: Your computer's muscle. `auto` is smart. (Default: `auto`) 💪
*   `--aligned`: Got pre-aligned 512x512 faces? This is for you! ✔️
*   `--ext [auto|png|jpg]`: How to save 'em. (Default: `auto`)
*   `--no-save-cropped / --no-save-restored / --no-save-comparison`: Less files, if you want. 🚫💾

**What Else Can It Do? Ask It!**
```bash
python inference_scalex.py --help
```

---

## 💡 Uh Oh! (Troubleshooting)
*   **`Module... functional_tensor` Error?!** 😵
    Relax! `patches.py` is our superhero for this. Should be automatic. If not, check it's there!
*   **Downloads Stuck?** 🐌
    Internet playing tricks? Delete the half-baked `.pth` file in `models/pretrained/` & try again.
*   **GPU Says "Nope!"?** 🙅‍♀️
    *   NVIDIA: Drivers up-to-date? PyTorch match your CUDA?
    *   Apple: Right macOS & PyTorch?
    *   Try `--device cpu`. If that works, it's a GPU thing.
*   **Install Script Blues?** 🎶
    Conda acting shy? Make sure it's in your PATH or `conda init your_shell_name` was run.
*   **Other `ModuleNotFoundError`s?** 🤷
    Are you *sure* you're in the `ScaleX` conda environment? Did `pip install -r requirements.txt` finish happily?

---

## 🤝 Join the Fun! (Contributing)
Got ideas? Found a 🐛? Want to add your own sparkle? ✨ Contributions are super welcome!
Check the [issues page](https://github.com/Md-Siam-Mia-Code/ScaleX/issues). Or just fork & send a Pull Request!

1.  🍴 Fork it!
2.  🌿 New Branch (`git checkout -b feature/SuperCoolThing`)
3.  💾 Commit (`git commit -m 'Added SuperCoolThing'`)
4.  🚀 Push (`git push origin feature/SuperCoolThing`)
5.  📬 Open Pull Request!

---

## 📜 The Fine Print (Acknowledgments & License)
ScaleX stands on the shoulders of giants! 🏋️‍♂️ Big virtual high-fives to:
*   **GFPGAN:** [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) - The OG face wizards!
*   **Real-ESRGAN:** [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - Background beautifiers!
*   **BasicSR & facexlib:** [xinntao](https://github.com/xinntao) - The building blocks!

---

# ❤️ *Go Make Some Pixels Happy!* pixels 😄
