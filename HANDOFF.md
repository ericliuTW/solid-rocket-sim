# 固態火箭推進概念模擬平台 — 設計哲學與技術報告 & 專案延續說明

> **文件用途**：AI Agent 交接文件，供接手開發的 AI（Codex 等）快速理解專案全貌。
> **撰寫日期**：2026-05-02
> **撰寫者**：Claude Opus 4.6（原開發 AI Agent）
> **專案擁有者**：yuanyuliu (GitHub: ericliuTW)

---

## 目錄

1. [專案概述](#1-專案概述)
2. [設計哲學](#2-設計哲學)
3. [架構與模組說明](#3-架構與模組說明)
4. [核心物理模型](#4-核心物理模型)
5. [已知問題與技術債](#5-已知問題與技術債)
6. [Bug 修復歷史與踩坑紀錄](#6-bug-修復歷史與踩坑紀錄)
7. [部署架構](#7-部署架構)
8. [開發環境](#8-開發環境)
9. [Git 工作流（重要）](#9-git-工作流重要)
10. [專案延續：待辦與建議方向](#10-專案延續待辦與建議方向)
11. [檔案清單與各模組職責](#11-檔案清單與各模組職責)
12. [給下一個 AI Agent 的注意事項](#12-給下一個-ai-agent-的注意事項)

---

## 1. 專案概述

**固態火箭推進概念模擬平台**（Solid Rocket Propulsion Conceptual Simulation Platform）是一個教學用工具，讓使用者透過互動介面探索固態火箭藥柱幾何對推進性能的影響。

### 核心功能
- 5 種藥柱幾何類型（BATES、圓柱、端面燃燒、星形、偏心孔）的燃燒面積曲線模擬
- 4 種噴嘴外型繪製（錐形、鐘形、直切喉管、雙錐角）— 僅影響繪圖
- 歸一化趨勢圖 + 示意估算值雙模式輸出
- 引擎縱剖面圖、橫截面圖、燃燒退化序列動畫
- ±5% 敏感度分析
- 教學用風險警告系統
- 1D 垂直飛行軌跡概念估算（含空氣阻力）
- 火箭總質量 vs 最大高度掃描曲線
- Streamlit Web UI + CLI 兩種使用方式
- 完整工程設計參考表輸出

### 定位（至關重要）
**本工具僅供教學與概念分析，絕不可用於真實推進器設計、製造或安全驗證。**
這不是免責聲明套話 — 模型確實忽略了侵蝕燃燒、溫度效應、多相流、點火暫態等重要現象。所有程式碼中的 `⚠` 標記和 disclaimer 都必須保留。

---

## 2. 設計哲學

### 2.1 教育優先，精確其次

整個系統的設計核心是「幫學生建立直覺」，而非「給出精確數值」。因此：

- **雙模式輸出**：歸一化模式（Normalized）讓學生看趨勢走向（漸進/中性/漸退），示意估算模式（Estimated）讓學生對「大概多少 MPa、多少 N」有量級直覺。
- **大量 disclaimer**：每個模組的 docstring、每張圖表的標題、報告的頭尾都有教學用途聲明。這是刻意的設計，不要刪除。
- **寧可保守也不誤導**：當模型的簡化假設可能導致數值不合理時（例如軌跡模組的推重比），寧可讓使用者自行輸入實際值（火箭總質量），也不要用可能嚴重偏差的自動估算。

### 2.2 視覺驅動的學習

這個專案遵循專案擁有者的教育軟體設計哲學：**「讓學生看到/操作/體驗」，文字只能輔助、不可主導**。因此：

- 引擎剖面圖是核心功能之一（不是附屬品）
- 燃燒退化序列讓學生「看到」藥柱怎麼燒
- 互動滑桿讓學生「操作」參數並立即看到結果
- 所有圖表都有顏色編碼和中文標註

### 2.3 CJK 字型處理策略

這是踩過很多坑才定下來的策略，**不要輕易改動**：

```
plotting.py::_find_cjk_font()
├── Windows: 掃描 C:/Windows/Fonts/msjh*.ttc (微軟正黑體)
├── Linux:   掃描 /usr/share/fonts/**/NotoSansCJK*
├── 找到後: fm.fontManager.addfont(path)  ← 關鍵！
└── 使用時: fontproperties=FontProperties(fname=path)  ← 不用 fontfamily！
```

**為什麼不用 `plt.rcParams['font.family']`？**
因為在 Streamlit Cloud 的 Linux 環境中，matplotlib 的字型快取不一定會即時更新。直接用 `FontProperties(fname=...)` 繞過快取，是最可靠的做法。

`engine_drawing.py` 從 `plotting.py` 匯入 `_cjk_font` 和 `_cjk_font_path`，統一字型來源。**不要讓 engine_drawing.py 有自己的字型偵測邏輯**，之前有過這個 bug。

### 2.4 噴嘴類型設計

噴嘴類型（`NozzleType`）**僅影響引擎剖面圖的繪製**，不改變模擬計算結果。這是有意的設計決策：
- 不同噴嘴的效率差異需要 CFD 或經驗修正係數，超出教學工具的範疇
- 在 UI 中已明確標示「僅影響繪圖」
- 4 種類型：Conical（直線收斂-擴散）、Bell（拋物線，用 sin + 二次曲線）、Straight-Cut（僅收斂）、Dual-Cone（大收斂角+小擴散角）

---

## 3. 架構與模組說明

```
solid-rocket-sim/
│
├── app.py              ← Streamlit Web UI 入口（607 行）
├── main.py             ← CLI 入口（227 行）
│
├── constants.py        ← 所有常數、枚舉、預設組態（186 行）
│   ├── GrainType       enum: BATES, CYLINDRICAL, END_BURNER, STAR, MOON_BURNER
│   ├── NozzleType      enum: CONICAL, BELL, STRAIGHT_CUT, DUAL_CONE
│   ├── GrainConfig     dataclass: 藥柱幾何參數
│   └── EXAMPLE_CONFIGS dict: 預設範例組態
│
├── geometry.py         ← 藥柱幾何計算（323 行）
│   ├── BurnProfile     dataclass: 燃燒面積/體積隨 web fraction 變化
│   └── GrainGeometry   class: compute_burn_profile() 分派到各類型
│
├── simulation.py       ← 概念模擬引擎（237 行）
│   ├── SimulationResult dataclass: 完整模擬結果
│   └── ConceptSimulator class: run() → 計算 Kn/壓力/推力時間序列
│
├── sensitivity.py      ← 敏感度分析（200 行）
│   └── SensitivityAnalyzer: ±5% 擾動各參數
│
├── risk_warnings.py    ← 教學用風險警告（364 行）
│   └── RiskFlagger: 檢查 Kn 範圍、壓力尖峰、推力穩定性等
│
├── trajectory.py       ← 飛行軌跡估算（306 行）
│   ├── RocketConfig    dataclass: 火箭整體參數
│   ├── TrajectoryResult dataclass: 軌跡估算結果
│   └── TrajectoryEstimator: 1D 垂直飛行 + 空氣阻力
│
├── plotting.py         ← matplotlib 圖表（479 行）
│   ├── _find_cjk_font() ← CJK 字型偵測（跨平台）
│   └── PlotManager: 單組/比較/敏感度圖表
│
├── engine_drawing.py   ← 引擎幾何繪圖（793 行）
│   └── EngineDrawing: 縱剖面 + 橫截面 + 燃燒序列 + 4 種噴嘴
│
├── report.py           ← 文字報告產生器（279 行）
│   └── generate_text_report(): 含工程設計參考表
│
├── .streamlit/config.toml  ← Streamlit 主題設定（深色主題）
├── packages.txt            ← Streamlit Cloud 的 apt 套件（fonts-noto-cjk）
├── requirements.txt        ← Python 依賴（numpy, matplotlib, streamlit）
├── Dockerfile              ← Docker 部署用
├── docker-compose.yml
├── DEPLOY.md               ← 8 種部署方式的完整指南
├── railway.toml / render.yaml / fly.toml / app.yaml / Procfile  ← 各平台部署設定
└── output/                 ← CLI 模式的輸出目錄
```

### 資料流

```
使用者輸入（Streamlit sidebar / CLI args）
    ↓
GrainConfig（constants.py）
    ↓
GrainGeometry.compute_burn_profile()（geometry.py）
    ↓ BurnProfile
ConceptSimulator.run()（simulation.py）
    ↓ SimulationResult
    ├──→ PlotManager（plotting.py）→ 圖表
    ├──→ EngineDrawing（engine_drawing.py）→ 引擎剖面圖
    ├──→ SensitivityAnalyzer（sensitivity.py）→ 敏感度報告
    ├──→ RiskFlagger（risk_warnings.py）→ 風險警告
    ├──→ TrajectoryEstimator（trajectory.py）→ 軌跡結果
    └──→ generate_text_report()（report.py）→ 文字報告
```

---

## 4. 核心物理模型

### 4.1 燃燒面積計算（geometry.py）

各類型藥柱的燃燒面積公式：

**BATES（最重要）**：
```
Ab = π·D_i·L·N + 2·(free_ends)·π/4·(D_o² - D_i²)
```
- `D_i`：內徑（隨燃燒回歸增大）
- `D_o`：外徑（固定）
- `L`：段長（隨端面燃燒縮短）
- `free_ends`：未被抑制的端面數 = `2*N - inhibited_ends*N`
- BATES 的特點：內徑燃燒 → 面積增大（漸進），端面燃燒 → 面積不變或減小，兩者可抵消達到「接近中性」

### 4.2 穩態壓力方程（simulation.py）

Saint-Venant 穩態假設：
```
P = (ρ · a · C* · η_c · Ab / At) ^ (1 / (1 - n))
```
- `ρ` = 1750 kg/m³（示意推進劑密度）
- `a` = 1.007e-4 m/s·Pa^(-n)（燃速係數，≈ KNSB 等級）
- `n` = 0.319（壓力指數）
- `C*` = 1550 m/s（特徵速度）
- `η_c` = 0.92（燃燒效率）
- `Ab`、`At`：燃燒面積、喉部面積（m²）

**⚠ 燃速係數 `a` 的單位至關重要**：必須是 SI 單位（m/s, Pa）。之前有過把 `a = 5.13e-3`（mm/s, MPa 單位系統）直接丟進 Pa 系統的 bug，導致壓力算出 12000 MPa 的荒謬值。正確轉換：`r[mm/s] = 8.26 · P[MPa]^0.319` → `a_SI = 8.26e-3 / (1e6)^0.319 = 1.007e-4`。

### 4.3 推力方程（simulation.py）

```
F = Cf · η_nozzle · P · At
```
- `Cf` = 1.45（推力係數）
- `η_nozzle` = 0.90（噴嘴效率）

### 4.4 軌跡模型（trajectory.py）

1D 垂直飛行，Euler 積分：
```
每個 dt：
  if t < burn_time:
    thrust = 插值(推力曲線, t)
    mass = m_total - (m_propellant · t / burn_time)  # 線性消耗
  else:
    thrust = 0
    mass = m_dry

  drag = 0.5 · Cd · A · ρ(h) · v²
  ρ(h) = ρ₀ · exp(-h / 8500)
  
  a = (thrust - drag) / mass - g
  v += a · dt
  h += v · dt
```

火箭總質量由使用者直接輸入（不再自動估算結構質量），乾重 = 總質量 - 推進劑質量。

---

## 5. 已知問題與技術債

### 高優先
1. **軌跡模組的推力插值精度**：目前用線性質量消耗假設，但推進劑消耗率其實隨壓力變化。對教學工具來說可接受，但如果要提升精度，應改用推力曲線積分來計算已消耗質量。
2. **Streamlit 頁面載入慢**：total payload-altitude 掃描曲線要跑 10 次完整軌跡模擬，在 Streamlit Cloud 上可能要幾秒。已從 20 點降到 10 點、dt 從 0.005 放大到 0.02。可考慮加 `@st.cache_data`。
3. **preview_screenshot 超時**：Streamlit 頁面太重，matplotlib 圖表太多，導致 preview 工具截圖超時。不影響實際使用，但開發時驗證不便。

### 中優先
4. **星形和偏心孔的幾何計算較粗糙**：目前是簡化版，真實星形需要更精確的多邊形面積計算。
5. **CLI 模式 (main.py) 尚未整合軌跡模組和噴嘴類型選擇**。
6. **沒有單元測試**：所有驗證都靠手動 + 目視檢查。
7. **Unicode 字元相容性**：`⚠` 和 `₀` 等符號在某些終端（Windows cp950）會 crash，plotting.py 中已替換為 `[!]` 和 `_0`，但 report.py 和 constants.py 中仍有 `⚠`。

### 低優先
8. **多語言支援**：目前 UI 全中文，沒有語言切換。
9. **圖表匯出功能**：使用者無法從 Streamlit UI 直接下載高解析度圖表。
10. **RLS/認證**：純前端應用，無後端資料儲存，無安全性問題。

---

## 6. Bug 修復歷史與踩坑紀錄

| 版本 | 問題 | 原因 | 修復方式 |
|------|------|------|---------|
| v1 | 壓力算出 12000 MPa | 燃速係數 `a` 用了 mm/s-MPa 單位值，但方程是 m/s-Pa | 正確轉換 `a = 1.007e-4` |
| v2 | Windows 上中文字變方框 | matplotlib 預設 DejaVu Sans 無 CJK 字形 | 加入 `_find_cjk_font()` 掃描系統字型 |
| v3 | `⚠` 和 `₀` 在 Windows 終端 crash | cp950 編碼不支援這些 Unicode 字元 | 圖表中替換為 ASCII 替代 |
| v4 | engine_drawing 中文也變方框 | engine_drawing.py 有自己的字型設定 `font.family = SimHei`，覆蓋了 plotting.py 的正確設定 | 改為從 plotting.py 匯入統一字型 |
| v5 | Streamlit Cloud 上中文方框 | Linux 無預裝 CJK 字型；即使 apt 安裝了，matplotlib 快取沒更新 | `packages.txt` + `addfont()` + `FontProperties(fname=...)` |
| v6 | 引擎剖面圖下半部上下顛倒 | 噴嘴繪製中 `sign * (R_i if sign > 0 else -R_o)` 的邏輯錯誤：sign=-1 時 `(-1)*(-R_o) = +R_o` | 改用明確的上半/下半 Rectangle 定義 |
| v7 | 切換預設組態時 StreamlitValueAboveMaxError | 預設 CD=25 但新 OD=20 → value > max_value | 所有 number_input 加 `min(max(...))` 夾限 |
| v8 | 軌跡模組推重比 145、速度 Mach 4 | 結構質量比 0.40 太低（不含機身） | 改為讓使用者直接輸入火箭總質量 |

### 踩坑教訓（給接手 AI 的提醒）

1. **matplotlib 字型問題**：不要用 `rcParams['font.family']` 設全域字型，要用 `FontProperties(fname=path)` 逐一指定。不要讓多個模組各自偵測字型。
2. **Streamlit number_input 值域**：當 max_value 動態變化時，預設值和當前值都要 clamp。Streamlit 會在 value > max_value 時直接 crash（不是 warning）。
3. **跨平台路徑**：字型路徑在 Windows 和 Linux 完全不同，必須用 glob 掃描。
4. **對稱繪圖**：用 `sign * value` 做上下對稱時，注意 sign=-1 會翻轉負值的符號。用明確的座標比 sign 乘法更安全。

---

## 7. 部署架構

### 目前部署狀態

| 環境 | 狀態 | 網址 |
|------|------|------|
| Local | ✅ 可用 | `http://localhost:8501` |
| Streamlit Cloud | ✅ 已部署 | 見 share.streamlit.io Dashboard |
| GitHub | ✅ public repo | `https://github.com/ericliuTW/solid-rocket-sim` |

### Streamlit Cloud 部署機制
- push 到 GitHub `master` branch → 自動重新部署
- `packages.txt` 中的 `fonts-noto-cjk` 會被 apt 安裝
- `requirements.txt` 中的 Python 套件會被 pip 安裝
- `.streamlit/config.toml` 設定深色主題

### 完整部署指南
見 `DEPLOY.md`，涵蓋 8 種部署方式，包含 Docker、Railway、Render、Fly.io、GCP Cloud Run 等。

---

## 8. 開發環境

| 項目 | 值 |
|------|-----|
| OS | Windows |
| Python | 3.8+（開發用 3.12 和 3.14 測試過）|
| Python 路徑 | `C:\AI38\Scripts\streamlit.exe`（本機 Streamlit） |
| 依賴 | numpy >= 1.24, matplotlib >= 3.7, streamlit >= 1.28 |
| 啟動指令 | `streamlit run app.py` 或 `python main.py` |
| launch.json | `.claude/launch.json` 設定了 Streamlit 預覽伺服器 |

### 本機啟動
```bash
cd D:/caludecode/solid-rocket-sim
C:\AI38\Scripts\streamlit.exe run app.py
```

### CLI 模式
```bash
cd D:/caludecode/solid-rocket-sim
C:\AI38\python.exe main.py --type bates --od 50 --length 70 --cd 18 --segments 4
```

---

## 9. Git 工作流（重要）

### ⚠ 嚴格規則

**AI Agent 絕不執行 `git add`、`git commit`、`git push` 或任何 git 操作。**

所有 Git 操作由專案擁有者透過 **GitHub Desktop** 手動進行。AI 的職責是：
1. 修改程式碼
2. 提供 Commit Summary 格式的改動摘要
3. 等用戶自行 commit 和 push

### Commit Summary 格式
```
📋 Commit Summary（YYYY-MM-DD）
一句話摘要

- 改動細節 1
- 改動細節 2
- ...
```

### 例外
如果專案有 go-ship skill（`.claude/skills/go-ship/`），用戶呼叫 `/go` 可視為授權本次 commit，但 push 仍需用戶明確說「可以 push」。本專案目前沒有 go-ship skill。

---

## 10. 專案延續：待辦與建議方向

### 10.1 短期待辦（bug fix / 優化）

- [ ] **加入 `@st.cache_data` 快取**：軌跡掃描曲線每次改動都重新計算 10 次，加快取可大幅改善頁面載入速度
- [ ] **CLI 整合軌跡模組**：`main.py` 尚未支援 trajectory 和 nozzle type
- [ ] **加入基本單元測試**：至少測 geometry.py 的面積計算和 simulation.py 的壓力計算
- [ ] **Unicode 清理**：report.py 和 constants.py 中的 `⚠` 在 Windows cp950 終端會 crash

### 10.2 中期功能建議

- [ ] **多推進劑配方比較**：讓使用者選擇不同的推進劑（KNSB、KNSU、APCP 等），各有不同的 `a`, `n`, `ρ`, `C*`
- [ ] **噴嘴效率修正**：根據噴嘴類型微調 `Cf`，讓噴嘴選擇也影響計算結果
- [ ] **圖表下載按鈕**：讓使用者從 UI 直接下載 PNG/SVG
- [ ] **馬達分級更準確**：目前的 K/L/M 分級依據 NAR 標準，但邊界值可以更精確
- [ ] **英文介面切換**：加語言選擇（中/英）

### 10.3 長期方向

- [ ] **2D 軸對稱模擬**：更精確的壓力場計算（需要 PDE solver）
- [ ] **多段不同幾何**：允許每段藥柱有不同的 OD/CD/L
- [ ] **結構分析**：根據壓力估算殼體壁厚需求（薄壁壓力容器公式）
- [ ] **3D 藥柱可視化**：用 Three.js 或 plotly 3D 取代 matplotlib 的 2D 剖面圖

---

## 11. 檔案清單與各模組職責

| 檔案 | 行數 | 職責 | 重要度 |
|------|------|------|--------|
| `app.py` | 607 | Streamlit Web UI，側邊欄控制、圖表顯示、軌跡區段 | ⭐⭐⭐ |
| `constants.py` | 186 | 所有枚舉、常數、預設參數、範例組態、免責聲明 | ⭐⭐⭐ |
| `geometry.py` | 323 | 藥柱幾何計算，5 種類型的燃燒面積曲線 | ⭐⭐⭐ |
| `simulation.py` | 237 | 核心模擬引擎，穩態壓力 + 推力計算 | ⭐⭐⭐ |
| `trajectory.py` | 306 | 1D 飛行軌跡估算，含空氣阻力模型 | ⭐⭐ |
| `engine_drawing.py` | 793 | 引擎剖面圖、橫截面、燃燒序列、4 種噴嘴繪製 | ⭐⭐ |
| `plotting.py` | 479 | matplotlib 圖表 + CJK 字型偵測（跨平台） | ⭐⭐ |
| `sensitivity.py` | 200 | ±5% 敏感度分析 | ⭐ |
| `risk_warnings.py` | 364 | 教學用風險警告（Kn、壓力、穩定性） | ⭐ |
| `report.py` | 279 | 文字報告 + 工程設計參考表 | ⭐ |
| `main.py` | 227 | CLI 入口（未整合軌跡模組） | ⭐ |
| `.streamlit/config.toml` | 16 | Streamlit 主題設定（深色琥珀主題） | 設定 |
| `packages.txt` | 1 | Streamlit Cloud apt 套件（CJK 字型） | 部署 |
| `requirements.txt` | 3 | Python 依賴 | 部署 |
| `Dockerfile` | — | Docker 部署 | 部署 |
| `DEPLOY.md` | 255 | 8 種部署方式的完整指南 | 文件 |

---

## 12. 給下一個 AI Agent 的注意事項

### 絕對不要做的事
1. **不要執行任何 git 操作**（commit, push, init, branch 等）
2. **不要刪除 disclaimer / 免責聲明文字**
3. **不要把 `plotting.py` 的字型偵測邏輯複製到其他模組** — 統一從 plotting.py 匯入
4. **不要用 `plt.rcParams['font.family']` 設定字型** — 用 `FontProperties(fname=path)`
5. **不要把燃速係數 `a` 的單位搞混**（SI: m/s + Pa，不是 mm/s + MPa）
6. **不要刪除資料庫資料**（雖然本專案沒有資料庫，但這是全域規則）

### 改動後必做的事
1. 在 app.py 中測試 Streamlit 是否正常啟動
2. 確認 number_input 的 value 不超過 max_value（會直接 crash）
3. 如果改了圖表，確認 CJK 中文字是否正常顯示
4. 提供 Commit Summary 給用戶

### 用戶偏好
- 繁體中文溝通
- 所有操作由 AI 執行，不要要求用戶指定檔案或行數
- 用白話文解釋技術概念
- 回應精簡，不做多餘的事後總結（總結統一放在 Commit Summary）

### 技術環境
- Windows 開發環境
- Python 3.12/3.14（`C:\AI38\`）
- Streamlit 預覽：`C:\AI38\Scripts\streamlit.exe run app.py`
- 部署在 Streamlit Cloud（GitHub auto-deploy）
- 專案路徑：`D:/caludecode/solid-rocket-sim/`

---

> 本文件由 Claude Opus 4.6 撰寫，基於完整的開發過程記憶。
> 如有疑問，可參考 git log 中的 commit 訊息或 `DEPLOY.md`。
