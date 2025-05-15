# 🩺 日本語 医療会話 NER アプリ

このプロジェクトは、日本語の医療会話テキストから固有表現（名前、地名、施設名など）を抽出するアプリケーションです。  
Hugging Face 上のファインチューニング済みモデルを利用し、Streamlit Cloud で公開しています。

---

## 🌐 公開アプリ

👉 [アプリを使ってみる](https://ftrobertaner001-6lycwuemlgun6zff5hgxmx.streamlit.app/)

---

## 📚 使用技術

- [Streamlit](https://streamlit.io/)  
- [Hugging Face Transformers](https://huggingface.co/)  
- [PyTorch](https://pytorch.org/)  

---

## 🚀 アプリの使い方

1. テキスト入力欄に日本語の医療会話文を入力します。  
2. 「解析実行」ボタンをクリックします。  
3. 抽出されたエンティティとそのラベル、信頼度が表示されます。

---

## 🏷️ 抽出可能なラベル一覧

| ラベル  | 説明        |
|---------|-------------|
| PER     | 人名         |
| ORG     | 組織名       |
| ORG-P   | 組織の部門   |
| ORG-O   | 組織その他   |
| LOC     | 地名         |
| INS     | 施設名（病院・学校など）|
| PRD     | 製品名       |
| EVT     | イベント名   |
| O       | その他（非エンティティ） |

---

## 📄 ライセンス

MIT License  
