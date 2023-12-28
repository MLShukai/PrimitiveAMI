# PrimitiveAMI

![pamiq-img](pamiq.png)

[\[デモ動画\]](https://youtu.be/5_ELaBQPLIY)

[\[論文\]](https://drive.google.com/drive/folders/1n_onR2X329P5y5ALWo2oZsJWSxeSjOnu?usp=sharing)

## 概要

VRChat上における好奇心ベースの原始自律機械知能の実装です。シンプルでかつこれからの土台となるAMIを実装しています。

### 著者

[<u>GesonAnko</u>](https://twitter.com/GesonAnkoVR),
[<u>myxy</u>](https://twitter.com/3405691582),
[<u>ocha_krg</u>](https://twitter.com/cehl_teapot),
[<u>ぶんちん</u>](https://twitter.com/bunnchinn3),
[<u>Klutz</u>](https://twitter.com/Earl_Klutz)

## 動作方法

### セットアップ

[PC環境のセットアップはこちらをご確認ください](https://confusion-universe-c5b.notion.site/Ubuntu22-04-b7b5f0be1abb40c2a60debae4f3e6043)

1. このリポジトリをクローンする
2. [`Miniforge3`](https://github.com/conda-forge/miniforge)を取得し、仮想環境を作成する。pythonのバージョンは3.10である。
3. [`pip install poetry`](https://python-poetry.org)でpoetryを取得し、このリポジトリ内で `poetry install` する。

### 起動

1. VRChatを起動する
2. OBSでVRChatをキャプチャし、仮想カメラを開始する。
3. 次のコマンドで起動する。
   ```sh
   python src/train.py
   ```

#### options

- 複数カメラデバイスがある時

  ```sh
  python src/train.py environment.sensor.camera_index=<index>
  ```

  NOTE: 事前にカメラデバイスのindexをメモしておく。

### docker

Dockerイメージをビルドして環境を構築する。

事前に必要な依存関係

- `docker`
- `make`
- `v4l-utils`

**NOTE: dockerイメージを起動する前にOBSを起動し、仮想カメラを有効化すること**

Dockerイメージをビルドし、起動する。

```sh
make docker-build
make docker-run
```

Dockerの起動時にOBSの仮想カメラを自動的に選択し、イメージ内で`/dev/video0`に接続している。(OpenCVのVideoCaptureのdevice indexが0になるように。)

Dockerはホストとネットワークを共有する、ホストモードで起動する。(VRChatのOSC APIを叩くため。) そのためネットワークポートの競合に注意が必要である。

# その他情報

- [VRChat自律機械知能プロジェクト中間発表その1](https://youtu.be/hwiJwuvRy9I)
- [VRChat自律機械知能プロジェクト中間発表その2](https://youtu.be/8-9rMswTZYs)
- [機械学習における「好奇心」](https://youtu.be/ACulPki98Ps)
- [入門：自律機械知能](https://youtu.be/P1LiB4WAIW4)
