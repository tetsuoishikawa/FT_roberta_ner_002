import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re

# モデルの読み込み
MODEL_REPO = "Tetsuo3003/ner-medical-japanese"
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForTokenClassification.from_pretrained(MODEL_REPO)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# ラベルごとの色設定
LABEL_COLORS = {
    "PER": "#FF6666",  # 赤系
    "ORG": "#66B2FF",  # 青系
    "LOC": "#66FF66",  # 緑系
    "INS": "#FFCC66",  # オレンジ系
    "PRD": "#CC99FF",  # 紫系
    "EVT": "#FF99CC",  # ピンク系
    "ORG-P": "#FFB266",
    "ORG-O": "#FFB266"
}

# 仮名加工関数（マスキング処理 + 色付きラベル）
def mask_entities(text, entities):
    masked_text = text
    for entity in sorted(entities, key=lambda x: -len(x['word'])):
        label = entity['entity_group']
        color = LABEL_COLORS.get(label, "#CCCCCC")  # デフォルト灰色
        replacement = f"<span style='color: {color}; font-weight: bold;'>[{label}]</span>"
        # エスケープ処理して置換（正規表現で単語全体一致）
        masked_text = re.sub(re.escape(entity['word']), replacement, masked_text)
    return masked_text

# アプリ UI
st.title("🩺 日本語 医療会話 NER アプリ（改良版・色付きラベル）")

# 入力フォーム
text = st.text_area("【①】解析したいテキストを入力してください（500文字まで）:", 
                    "金丸先生が松本市にある石川クリニックに通院しました。", max_chars=500)

# 解析ボタン
if st.button("【②】解析開始"):
    with st.spinner("解析中..."):
        results = ner_pipeline(text)
        
        # 仮名加工（マスキング＋色付き）
        masked_text = mask_entities(text, results)

        # 結果表示
        st.subheader("【③a】📝 仮名加工（マスキング）後の文章")
        st.markdown(masked_text, unsafe_allow_html=True)

        st.subheader("【③b】🔍 抽出したエンティティ一覧")
        if results:
            for entity in results:
                st.write(f"- **{entity['word']}** → {entity['entity_group']} (信頼度: {entity['score']:.2f})")
        else:
            st.info("エンティティは検出されませんでした。")
