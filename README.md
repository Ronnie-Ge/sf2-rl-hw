## Street Fighter II 強化學習專案

這個專案是以 `Street Fighter II: Special Champion Edition` 為題目的強化學習實作，使用：

- `uv` 管理 Python 環境與依賴
- `gym-retro` 建立遊戲環境
- `stable-baselines3` 的 `PPO` 訓練 agent
- `ffmpeg` 將 rollout frame 編碼成 `.mp4`

目前專案已支援：

- `train`：訓練模型，定期儲存 checkpoint，並輸出 evaluation metrics
- `evaluate`：讀取指定 checkpoint，輸出評估結果
- `record`：讀取單一或多個 checkpoint，產生帶 overlay 的影片
- `YAML config + extends` 的實驗設定管理
- `baseline` 與 `reference_v1` reward profile

預設 PPO 設定已調整為較保守、較適合約 `8GB VRAM` 顯卡的版本。

## 目錄結構

```text
sf2-rl-hw/
├── configs/
│   ├── base.yaml
│   └── experiments/
│       └── baseline.yaml
├── retro_data/
│   └── StreetFighterIISpecialChampionEdition-Genesis/
│       ├── Champion.Level12.RyuVsBison.state
│       ├── data.json
│       ├── metadata.json
│       └── scenario.json
├── src/sf2_rl_hw/
│   ├── cli.py
│   ├── config.py
│   ├── train.py
│   ├── evaluate.py
│   ├── record.py
│   ├── rollout.py
│   ├── agents/
│   ├── envs/
│   ├── rewards/
│   └── utils/
├── pyproject.toml
├── uv.lock
└── README.md
```

## 環境需求

### Python

- 專案預設 Python 版本為 `3.8`
- 根目錄的 `.python-version` 目前為 `3.8`

### 系統依賴

你至少需要：

- `ffmpeg`
- 可以執行 `gym-retro` 的系統環境

`ffmpeg` 用途：

- `record` 會先輸出 frame，再透過 `ffmpeg` 編碼成 `.mp4`
- 如果沒有 `ffmpeg`，訓練仍可進行，但影片無法產生

檢查方式：

```bash
ffmpeg -version
```

如果能看到版本資訊，表示已安裝完成。

### Python 套件依賴

主要依賴定義在 `pyproject.toml`：

- `gym-retro`
- `stable-baselines3`
- `torch`
- `numpy`
- `Pillow`
- `PyYAML`

## 專案建置

### 1. 進入專案目錄

```bash
cd sf2-rl-hw
```

### 2. 使用 `uv` 建立虛擬環境並安裝依賴

```bash
uv sync
```

這個指令會：

- 建立 `.venv/`
- 根據 `uv.lock` 安裝依賴
- 安裝 CLI 指令 `sf2-rl-hw`

### 3. 驗證 CLI

```bash
uv run sf2-rl-hw --help
```

如果安裝成功，會看到 `train / evaluate / record` 三個子指令。

## 必要檔案與資料

### 專案已提供的檔案

專案內已經包含 `gym-retro` integration 需要的這些檔案：

- `retro_data/StreetFighterIISpecialChampionEdition-Genesis/Champion.Level12.RyuVsBison.state`
- `retro_data/StreetFighterIISpecialChampionEdition-Genesis/data.json`
- `retro_data/StreetFighterIISpecialChampionEdition-Genesis/metadata.json`
- `retro_data/StreetFighterIISpecialChampionEdition-Genesis/scenario.json`

這些檔案用來：

- 指定遊戲起始 state
- 定義 `agent_hp / enemy_hp` 等記憶體映射
- 提供 reward 與 overlay 需要的遊戲資訊

### 你需要自行準備的檔案

你需要自行準備合法取得的遊戲 ROM。

目前預設 config 期待的 ROM 路徑是：

```text
../ROM/Street Fighter II' - Special Champion Edition (USA).zip
```

也就是相對於 `sf2-rl-hw/`，預期 ROM 放在：

```text
../ROM/Street Fighter II' - Special Champion Edition (USA).zip
```

### ROM 可以是 zip 還是解壓後的檔案？

兩種都可以：

- 如果 `rom_path` 指向 `.zip`
  - 專案會在第一次建環境時自動解壓出 ROM，放成 `retro_data/.../rom.md`
- 如果 `rom_path` 指向已解壓的 ROM 檔
  - 專案會直接複製或建立連結成 `retro_data/.../rom.md`

### `rom.md` 需要手動準備嗎？

不用。

`rom.md` 是執行時產物，程式會自動建立，不需要手動改名。

## 設定檔說明

專案使用 `configs/` 管理實驗設定：

- `configs/base.yaml`：共通預設值
- `configs/experiments/baseline.yaml`：實驗設定，透過 `extends` 繼承 `base.yaml`

實際載入邏輯在 `src/sf2_rl_hw/config.py`。

### 目前預設 baseline 設定

重要項目如下：

- `runtime.device: auto`
- `env.frame_skip: 6`
- `env.frame_stack: 9`
- `env.width: 128`
- `env.height: 100`
- `ppo.num_envs: 4`
- `ppo.n_steps: 256`
- `ppo.batch_size: 256`
- `ppo.checkpoint_freq: 500000`
- `evaluation.episodes: 5`
- `recording.episodes: 1`

### 如何修改 ROM 路徑

有兩種方式。

方式一：直接改 config

修改 `configs/base.yaml` 的：

```yaml
env:
  rom_path: /absolute/path/to/your/rom/or/archive
```

方式二：用環境變數覆蓋

```bash
export SF2_ROM_PATH="/absolute/path/to/your/rom/or/archive"
```

### 如何新增自己的實驗設定

建議不要直接一直改 `base.yaml`，而是在 `configs/experiments/` 新增新的 yaml。

範例：

```yaml
extends: ../base.yaml

name: debug-small

runtime:
  experiment_name: debug-small

ppo:
  total_timesteps: 20000
  num_envs: 1
  n_steps: 64
  batch_size: 64
  n_epochs: 1

evaluation:
  episodes: 1
```

然後用：

```bash
uv run sf2-rl-hw train --config configs/experiments/debug-small.yaml
```

## 如何開始訓練

### 最基本的訓練指令

```bash
uv run sf2-rl-hw train --config configs/experiments/baseline.yaml
```

這個流程會：

- 讀取 config
- 準備 ROM 與 `retro_data`
- 建立 `gym-retro` 環境
- 建立 PPO 模型
- 開始訓練
- 定期儲存 checkpoint
- 定期輸出 evaluation metrics

### 訓練輸出位置

所有輸出都會寫到 `artifacts/`，包含：

- `artifacts/checkpoints/<experiment>/<run>/`
- `artifacts/eval/<experiment>/<run>/`
- `artifacts/logs/<experiment>/<run>/`
- `artifacts/runs/<experiment>/<run>/`

重點檔案通常包括：

- `resolved_config.json`
- `run_metadata.json`
- `ppo_step_500000.zip`
- `metrics.json`
- `episode_metrics.json`

## 如何評估模型

### 指定 checkpoint 評估

```bash
uv run sf2-rl-hw evaluate \
  --config configs/experiments/baseline.yaml \
  --checkpoint /path/to/ppo_step_500000.zip
```

### 不指定 checkpoint

如果不給 `--checkpoint`，程式會嘗試自動找該實驗最新的 checkpoint。

```bash
uv run sf2-rl-hw evaluate --config configs/experiments/baseline.yaml
```

### 評估輸出

評估會輸出：

- `metrics.json`
- `episode_metrics.json`

主要指標包含：

- `win_rate`
- `mean_episode_return`
- `mean_final_hp_diff`
- `mean_episode_length`

## 如何錄影

### 單一 checkpoint 錄影

```bash
uv run sf2-rl-hw record \
  --config configs/experiments/baseline.yaml \
  --checkpoint /path/to/ppo_step_500000.zip
```

### 錄最新的 N 個 checkpoint

```bash
uv run sf2-rl-hw record \
  --config configs/experiments/baseline.yaml \
  --latest 3
```

### 用 glob 批次錄影

```bash
uv run sf2-rl-hw record \
  --config configs/experiments/baseline.yaml \
  --glob "artifacts/checkpoints/baseline/**/*.zip"
```

### 錄影輸出內容

錄影會輸出到：

- `artifacts/videos/<experiment>/<run>/`

內容包含：

- `.mp4` 影片
- `record_metrics.json`
- `batch_record_manifest.json`（batch 模式）

### 影片 overlay 會顯示哪些資訊

目前至少包含：

- `experiment_name`
- `checkpoint_step`
- `episode`
- `env_step`
- `action`
- `instant_reward`
- `episode_return`
- `agent_hp`
- `enemy_hp`
- `result`

## GPU 使用方式

目前預設為：

```yaml
runtime:
  device: auto
```

表示：

- 如果 PyTorch 偵測到可用 GPU，PPO 會使用 GPU
- 如果沒有可用 GPU，則回退到 CPU

如果你想強制使用 CPU：

```yaml
runtime:
  device: cpu
```

如果你想強制使用 CUDA：

```yaml
runtime:
  device: cuda
```

注意：

- GPU 主要用在 PPO 模型訓練與推論
- `gym-retro` 環境本身、frame 處理、`ffmpeg` 編碼不會因此全部改成 GPU 執行

## 推薦操作順序

建議照這個順序：

1. 安裝 `ffmpeg`
2. 放好 ROM，或設定 `SF2_ROM_PATH`
3. 執行 `uv sync`
4. 先跑一次 CLI help

```bash
uv run sf2-rl-hw --help
```

5. 先做小型訓練 smoke test
6. 確認有 checkpoint 後，再跑 `evaluate`
7. 最後再跑 `record`

## 常見問題

### 1. `ffmpeg` 找不到

現象：

- `record` 失敗
- 錯誤訊息提到 `ffmpeg executable not found`

處理方式：

- 先確認 `ffmpeg -version`
- 確保 `ffmpeg` 在 shell 的 `PATH` 內

### 2. 找不到 ROM

現象：

- 錯誤訊息提到 `Configured ROM path does not exist`

處理方式：

- 檢查 `configs/base.yaml` 的 `env.rom_path`
- 或重新設定：

```bash
export SF2_ROM_PATH="/absolute/path/to/your/rom/or/archive"
```

### 3. `gym` 的 legacy warning

現象：

- 看到 `Gym has been unmaintained since 2022 ...`

說明：

- 這是 `gym-retro` 與 `stable-baselines3` 相容層產生的警告
- 目前屬於已知現象，不代表訓練失敗

### 4. 沒有 checkpoint 可評估或錄影

現象：

- `evaluate` 或 `record` 提示找不到 checkpoint

處理方式：

- 先確認是否已跑過 `train`
- 檢查 `artifacts/checkpoints/` 下是否有 `.zip`
- 或手動用 `--checkpoint` 指定完整路徑

## 補充說明

- `retro_data/.../rom.md` 是執行時產物，不需要手動建立
- `record` 若遇到影片編碼問題，會保留 `.ffmpeg.log`
- `record` 的 batch 模式會輸出 `batch_record_manifest.json`，方便整理各 checkpoint 的影片結果
