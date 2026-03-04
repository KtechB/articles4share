# Articles4Share

各種資料について share するためのレポジトリ

NOTICE: 本リポジトリの内容は無断利用不可（All rights reserved）です。詳細は `LICENSE` を参照してください。

## ディレクトリ構成

- `articles/<article_id>/`: 記事・スライド一式
  - 例: `index.md`, `deck.md`, `deck.html`, `deck.pdf`, `deck.pptx`, `assets/`
- `slides_pdf/<article_id>.pdf`: 共有用の PDF
- `themes/`: 共有物の閲覧や再出力で使うテーマ資産

## 運用ルール

「Published に入れてください」と依頼された資料は、以下の形で配置します。

- 元資料一式は `publish/articles4share/articles/<article_id>/` にまとめる
- 共有用 PDF は `publish/articles4share/slides_pdf/<article_id>.pdf` に置く
- `<article_id>` は記事/スライドのディレクトリ名をそのまま使う
- 原文の article を含めるかは指示ベース。指示がない限りは不要

## 例

- `publish/articles4share/articles/20260222-paper-reading-SELF-DISTILLATION-ENABLES-CONTINUAL-LEARNING/`
- `publish/articles4share/slides_pdf/20260222-paper-reading-SELF-DISTILLATION-ENABLES-CONTINUAL-LEARNING.pdf`
